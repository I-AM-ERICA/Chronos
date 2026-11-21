from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ExperimentConfig, PROJECT_ROOT
from .data import split_target
from .metrics import calculate_metrics
from .online_retail import load_online_retail_revenue


LAGS = (1, 2, 3, 7, 14, 21, 28, 56)
ROLLING_WINDOWS = (7, 14, 28, 56)
MAX_HISTORY = max(LAGS + ROLLING_WINDOWS)
SUPPORTED_MODELS = ("random_forest", "xgboost", "catboost")


def _feature_row(history: np.ndarray, date: pd.Timestamp) -> dict[str, float]:
    values = np.asarray(history, dtype=np.float64)
    if len(values) < MAX_HISTORY:
        raise ValueError(f"At least {MAX_HISTORY} historical days are required.")
    row: dict[str, float] = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = float(values[-lag])
    for window in ROLLING_WINDOWS:
        window_values = values[-window:]
        row[f"rolling_mean_{window}"] = float(np.mean(window_values))
        row[f"rolling_std_{window}"] = float(np.std(window_values, ddof=0))
        row[f"rolling_min_{window}"] = float(np.min(window_values))
        row[f"rolling_max_{window}"] = float(np.max(window_values))
    row["lag_1_minus_7"] = row["lag_1"] - row["lag_7"]
    row["mean_7_minus_28"] = row["rolling_mean_7"] - row["rolling_mean_28"]
    row["day_of_week"] = float(date.dayofweek)
    row["day_of_month"] = float(date.day)
    row["day_of_year"] = float(date.dayofyear)
    row["week_of_year"] = float(date.isocalendar().week)
    row["month"] = float(date.month)
    row["is_weekend"] = float(date.dayofweek >= 5)
    return row


def build_supervised_rows(
    values: np.ndarray, dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build one-step rows whose features use only earlier observations."""
    values = np.asarray(values, dtype=np.float64)
    dates = pd.DatetimeIndex(dates)
    if len(values) != len(dates):
        raise ValueError("values and dates must have the same length")
    if len(values) <= MAX_HISTORY:
        raise ValueError(f"More than {MAX_HISTORY} observations are required.")
    feature_rows = []
    targets = []
    target_indices = []
    for target_index in range(MAX_HISTORY, len(values)):
        feature_rows.append(_feature_row(values[:target_index], dates[target_index]))
        targets.append(values[target_index])
        target_indices.append(target_index)
    return (
        pd.DataFrame(feature_rows),
        np.asarray(targets, dtype=np.float64),
        np.asarray(target_indices, dtype=np.int64),
    )


def _make_model(name: str, seed: int, params: dict | None = None):
    params = dict(params or {})
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        defaults: dict[str, object] = dict(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        )
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError(
                "xgboost is required. Install it with: pip install xgboost"
            ) from error
        defaults = dict(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        defaults.update(params)
        return XGBRegressor(**defaults)
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise ImportError(
                "catboost is required. Install it with: pip install catboost"
            ) from error
        defaults = dict(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            thread_count=-1,
        )
        defaults.update(params)
        return CatBoostRegressor(**defaults)
    raise ValueError(f"Unsupported model {name!r}; choose from {SUPPORTED_MODELS}")


def _fit_model_and_residuals(
    name: str,
    train_values: np.ndarray,
    train_dates: pd.DatetimeIndex,
    seed: int,
    params: dict | None = None,
) -> tuple[Any, np.ndarray, dict[str, int]]:
    features, targets, target_indices = build_supervised_rows(train_values, train_dates)
    calibration_days = min(56, max(28, len(targets) // 5))
    calibration_start = len(train_values) - calibration_days
    fit_mask = target_indices < calibration_start
    if int(fit_mask.sum()) < 30:
        raise ValueError("Training history is too short for residual calibration.")

    calibration_model = _make_model(name, seed, params)
    calibration_model.fit(features.loc[fit_mask], targets[fit_mask])
    calibration_predictions = calibration_model.predict(features.loc[~fit_mask])
    residuals: np.ndarray = targets[~fit_mask] - np.asarray(
        calibration_predictions, dtype=np.float64
    )
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 10:
        raise ValueError("Too few finite calibration residuals were produced.")

    final_model = _make_model(name, seed, params)
    final_model.fit(features, targets)
    audit = {
        "train_observations": len(train_values),
        "supervised_rows": len(targets),
        "first_target_index": int(target_indices.min()),
        "last_target_index": int(target_indices.max()),
        "calibration_days": int(calibration_days),
    }
    return final_model, residuals, audit


def recursive_samples(
    model,
    context: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    residuals: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Create recursive paths without reading any future actual value."""
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    context = np.asarray(context, dtype=np.float64)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    probabilities = (np.arange(num_samples, dtype=np.float64) + 0.5) / num_samples
    residual_grid = np.quantile(np.asarray(residuals, dtype=np.float64), probabilities)
    histories = [context.copy() for _ in range(num_samples)]
    paths = np.empty((num_samples, len(forecast_dates)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for step, date in enumerate(forecast_dates):
        step_features = pd.DataFrame(
            [_feature_row(history, date) for history in histories]
        )
        predictions = np.asarray(model.predict(step_features), dtype=np.float64)
        innovations = residual_grid[rng.permutation(num_samples)]
        step_values = predictions + innovations
        paths[:, step] = step_values
        for sample_index in range(num_samples):
            histories[sample_index] = np.append(
                histories[sample_index], step_values[sample_index]
            )
    return paths


def _forecast_frame(
    model_name: str,
    split_name: str,
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    samples: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "split": split_name,
            "date": [date.date().isoformat() for date in dates],
            "actual_revenue": actual,
            "pred_q10": np.quantile(samples, 0.1, axis=0),
            "pred_q50": np.quantile(samples, 0.5, axis=0),
            "pred_q90": np.quantile(samples, 0.9, axis=0),
        }
    )


def run_tabular_benchmarks(
    data_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    model_names: Iterable[str] = SUPPORTED_MODELS,
) -> pd.DataFrame:
    """Fit tabular baselines on the Chronos train prefix and score the same holdouts."""
    requested = list(dict.fromkeys(model_names))
    unknown = sorted(set(requested) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    if not requested:
        raise ValueError("At least one model must be requested.")

    dataset = load_online_retail_revenue(data_path)
    row = dataset[0]
    target = np.asarray(row["target"], dtype=np.float64)
    dates = pd.DatetimeIndex(row["timestamp"])
    split = split_target(target, config)
    train_length = len(split.train)
    validation_dates = dates[train_length : train_length + config.validation_length]
    test_dates = dates[-config.prediction_length :]
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    forecast_frames = []
    model_audits: dict[str, dict[str, int]] = {}
    audit_payload: dict[str, object] = {
        "data_path": str(Path(data_path).resolve()),
        "total_observations": len(target),
        "train_observations": train_length,
        "validation_observations": config.validation_length,
        "test_observations": config.prediction_length,
        "feature_lags": list(LAGS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "models": model_audits,
    }
    for model_index, model_name in enumerate(requested):
        model, residuals, model_audit = _fit_model_and_residuals(
            model_name, split.train, dates[:train_length], config.seed
        )
        model_audits[model_name] = model_audit
        for split_index, (
            split_name,
            context,
            actual,
            forecast_dates,
        ) in enumerate(
            [
                (
                    "validation",
                    split.validation_context,
                    split.validation_truth,
                    validation_dates,
                ),
                ("test", split.test_context, split.test_truth, test_dates),
            ]
        ):
            samples = recursive_samples(
                model,
                context,
                forecast_dates,
                residuals,
                config.num_samples,
                config.seed + model_index * 100 + split_index,
            )
            metrics = calculate_metrics(
                samples[None, :, :],
                np.asarray(actual, dtype=np.float64)[None, :],
                [np.asarray(context, dtype=np.float64)],
                config.seasonal_period,
            )
            metric_rows.append(
                {
                    "dataset": "online-retail-revenue",
                    "split": split_name,
                    "model": model_name,
                    "label": model_name,
                    "series_count": 1,
                    **metrics,
                }
            )
            forecast_frames.append(
                _forecast_frame(
                    model_name,
                    split_name,
                    forecast_dates,
                    np.asarray(actual, dtype=np.float64),
                    samples,
                )
            )
            np.savez_compressed(
                output_dir / f"{model_name}_{split_name}_predictions.npz",
                samples=samples[None, :, :].astype(np.float32),
                actual=np.asarray(actual, dtype=np.float32)[None, :],
                series_ids=np.asarray([row["id"]]),
            )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "tabular_model_metrics.csv", index=False)
    pd.concat(forecast_frames, ignore_index=True).to_csv(
        output_dir / "tabular_model_forecasts.csv", index=False
    )
    (output_dir / "benchmark_audit.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    test_metrics = metrics_frame.loc[metrics_frame["split"] == "test"].copy()
    chronos_comparison_path = (
        PROJECT_ROOT
        / "outputs"
        / "online_retail_revenue"
        / "evaluation"
        / "comparison_metrics.csv"
    )
    if chronos_comparison_path.exists():
        chronos_metrics = pd.read_csv(chronos_comparison_path)
        chronos_metrics = chronos_metrics.loc[chronos_metrics["split"] == "test"]
        comparison = pd.concat([chronos_metrics, test_metrics], ignore_index=True)
    else:
        comparison = test_metrics
    comparison = comparison.sort_values(["WAPE", "MAE"], na_position="last")
    comparison.to_csv(output_dir / "all_model_comparison.csv", index=False)
    return comparison
