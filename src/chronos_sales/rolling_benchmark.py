from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig, PROJECT_ROOT
from .forecasting import predict_samples
from .metrics import QUANTILES, calculate_metrics, calculate_quantile_metrics
from .online_retail import load_online_retail_revenue
from .revenue_evaluation import seasonal_naive_samples
from .tabular_baselines import (
    LAGS,
    ROLLING_WINDOWS,
    _feature_row,
    _fit_model_and_residuals,
    _make_model,
    recursive_samples,
)
from .training import find_latest_checkpoint, run_finetuning


RANKED_MODELS = (
    "chronos_finetuned",
    "random_forest",
    "xgboost",
    "catboost",
    "auto_arima",
    "auto_ets",
)
REFERENCE_MODELS = ("seasonal_naive_7", "chronos_zero_shot")
EXPECTED_FOLD3 = {"WAPE": 0.21228311522801027, "MASE": 0.9583043721937496}
STAT_LEVELS = [20, 40, 60, 80]


TREE_CANDIDATES: dict[str, list[dict]] = {
    "random_forest": [
        {"max_depth": 8, "min_samples_leaf": 3, "max_features": 0.8},
        {"max_depth": 12, "min_samples_leaf": 3, "max_features": 0.8},
        {"max_depth": None, "min_samples_leaf": 3, "max_features": 0.8},
        {"max_depth": 12, "min_samples_leaf": 1, "max_features": 1.0},
    ],
    "xgboost": [
        {"max_depth": 3, "learning_rate": 0.03},
        {"max_depth": 4, "learning_rate": 0.03},
        {"max_depth": 6, "learning_rate": 0.03},
        {"max_depth": 4, "learning_rate": 0.05},
    ],
    "catboost": [
        {"depth": 4, "learning_rate": 0.03},
        {"depth": 6, "learning_rate": 0.03},
        {"depth": 8, "learning_rate": 0.03},
        {"depth": 6, "learning_rate": 0.05},
    ],
}


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    test_start_date: str
    test_end_date: str

    @property
    def name(self) -> str:
        return f"fold_{self.fold:02d}"


def make_fold_specs(
    dates: pd.DatetimeIndex, config: ExperimentConfig, folds: int = 3
) -> list[FoldSpec]:
    dates = pd.DatetimeIndex(dates)
    if folds < 1:
        raise ValueError("folds must be at least 1")
    needed = config.validation_length + folds * config.prediction_length
    if len(dates) <= needed:
        raise ValueError("The series is too short for the requested rolling folds.")
    specs = []
    for offset in range(folds):
        test_end = len(dates) - (folds - offset - 1) * config.prediction_length
        test_start = test_end - config.prediction_length
        validation_end = test_start
        validation_start = validation_end - config.validation_length
        train_end = validation_start
        specs.append(
            FoldSpec(
                fold=offset + 1,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
                train_end_date=dates[train_end - 1].date().isoformat(),
                validation_start_date=dates[validation_start].date().isoformat(),
                validation_end_date=dates[validation_end - 1].date().isoformat(),
                test_start_date=dates[test_start].date().isoformat(),
                test_end_date=dates[test_end - 1].date().isoformat(),
            )
        )
    return specs


def _fold_arrays(target: np.ndarray, spec: FoldSpec) -> dict[str, np.ndarray]:
    values = np.asarray(target, dtype=np.float64)
    return {
        "train": values[: spec.train_end],
        "validation_context": values[: spec.validation_start],
        "validation_truth": values[spec.validation_start : spec.validation_end],
        "test_context": values[: spec.test_start],
        "test_truth": values[spec.test_start : spec.test_end],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    names = ["chronos-forecasting", "torch", "scikit-learn", "xgboost", "catboost", "statsforecast"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _quantiles_from_samples(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("samples must have shape [samples, horizon]")
    return np.quantile(values, QUANTILES, axis=0)


def _metrics_from_quantiles(
    quantiles: np.ndarray, actual: np.ndarray, context: np.ndarray, config: ExperimentConfig
) -> dict[str, float]:
    return calculate_quantile_metrics(
        np.asarray(quantiles, dtype=np.float64)[None, :, :],
        np.asarray(actual, dtype=np.float64)[None, :],
        [np.asarray(context, dtype=np.float64)],
        config.seasonal_period,
    )


def _prediction_rows(
    spec: FoldSpec,
    model: str,
    split: str,
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    quantiles: np.ndarray,
) -> list[dict]:
    rows = []
    for day_index, date in enumerate(pd.DatetimeIndex(dates)):
        row = {
            "fold": spec.fold,
            "model": model,
            "split": split,
            "date": date.date().isoformat(),
            "actual_revenue": float(actual[day_index]),
        }
        for quantile_index, quantile in enumerate(QUANTILES):
            row[f"pred_q{int(quantile * 100):02d}"] = float(
                quantiles[quantile_index, day_index]
            )
        rows.append(row)
    return rows


def _metric_row(
    spec: FoldSpec, model: str, split: str, metrics: dict[str, float]
) -> dict:
    return {
        "fold": spec.fold,
        "model": model,
        "split": split,
        "train_end": spec.train_end_date,
        "validation_start": spec.validation_start_date,
        "validation_end": spec.validation_end_date,
        "test_start": spec.test_start_date,
        "test_end": spec.test_end_date,
        **metrics,
    }


def _write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def prepare_rolling_benchmark(
    data_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    folds: int = 3,
) -> list[FoldSpec]:
    from gluonts.dataset.arrow import ArrowWriter

    dataset = load_online_retail_revenue(data_path)
    row = dataset[0]
    target = np.asarray(row["target"], dtype=np.float64)
    dates = pd.DatetimeIndex(row["timestamp"])
    specs = make_fold_specs(dates, config, folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for spec in specs:
        fold_dir = output_dir / "folds" / spec.name
        arrow_dir = fold_dir / "arrow" / "train"
        arrow_path = arrow_dir / "data.arrow"
        if not arrow_path.exists():
            arrow_dir.mkdir(parents=True, exist_ok=True)
            ArrowWriter(compression="lz4").write_to_file(
                iter(
                    [
                        {
                            "start": np.datetime64(dates[0]),
                            "target": target[: spec.train_end].astype(np.float32),
                        }
                    ]
                ),
                arrow_path,
            )
        arrow_manifest_path = fold_dir / "arrow_manifest.json"
        arrow_manifest = {
            "train_length": spec.train_end,
            "train_end_date": spec.train_end_date,
            "arrow_path": str(arrow_path.resolve()),
            "sha256": _sha256(arrow_path),
        }
        if arrow_manifest_path.exists():
            recorded = json.loads(arrow_manifest_path.read_text(encoding="utf-8"))
            if recorded["train_length"] != spec.train_end or recorded["sha256"] != arrow_manifest["sha256"]:
                raise ValueError(f"Existing Arrow data does not match {spec.name}.")
        else:
            arrow_manifest_path.write_text(
                json.dumps(arrow_manifest, indent=2), encoding="utf-8"
            )
        if len(target[: spec.train_end]) != spec.train_end:
            raise AssertionError("Fold train length does not match its Arrow prefix.")
        fold_rows.append(asdict(spec))
        (fold_dir / "fold_spec.json").write_text(
            json.dumps(asdict(spec), indent=2), encoding="utf-8"
        )
    pd.DataFrame(fold_rows).to_csv(output_dir / "folds.csv", index=False)
    manifest = {
        "dataset": "online-retail-revenue",
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": _sha256(Path(data_path)),
        "fold_count": folds,
        "folds": fold_rows,
        "config": asdict(config),
        "versions": _versions(),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return specs


def _load_specs(output_dir: Path) -> list[FoldSpec]:
    path = output_dir / "folds.csv"
    if not path.exists():
        raise FileNotFoundError("Run the prepare stage before this stage.")
    return [FoldSpec(**row) for row in pd.read_csv(path).to_dict("records")]


def _checkpoint_candidates(training_root: Path) -> list[Path]:
    candidates = []
    for path in training_root.glob("run-*/checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit() and (path / "config.json").exists():
            candidates.append(path)
    return sorted(candidates, key=lambda item: int(item.name.rsplit("-", 1)[-1]))


def _write_chronos_marker(fold_dir: Path, checkpoint: Path, reused: bool) -> None:
    marker_dir = fold_dir / "models" / "chronos"
    marker_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = marker_dir / "checkpoint-selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    source_files = [path for path in checkpoint.iterdir() if path.is_file()]
    for source in source_files:
        destination = selected_dir / source.name
        if destination.exists():
            if _sha256(destination) != _sha256(source):
                raise ValueError(f"Saved Chronos artifact differs from {source}.")
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    (marker_dir / "selected_checkpoint.txt").write_text(
        str(selected_dir.resolve()), encoding="utf-8"
    )
    checkpoint_files = [path for path in selected_dir.iterdir() if path.is_file()]
    selection = {
        "checkpoint": str(selected_dir.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
        "reused_existing_fold3": reused,
        "artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(checkpoint_files)
        ],
    }
    selection_path = marker_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    manifest_path = fold_dir / "models" / "model_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"selected_settings": {}, "artifacts": {}, "versions": _versions()}
    )
    manifest["selected_settings"]["chronos_finetuned"] = {
        "checkpoint": str(checkpoint.resolve()),
        "selection_metric": "validation_WQL",
        "reused_existing_fold3": reused,
    }
    manifest["artifacts"]["chronos_finetuned"] = selection["artifacts"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _find_fold3_reuse() -> tuple[Path, Path, Path] | None:
    candidates = [
        (
            PROJECT_ROOT / "outputs" / "gpu-server" / "checkpoint-500",
            PROJECT_ROOT / "outputs" / "gpu-server" / "evaluation",
            PROJECT_ROOT / "outputs" / "gpu-server" / "zero_shot",
        ),
        (
            PROJECT_ROOT
            / "outputs"
            / "online_retail_revenue"
            / "finetune"
            / "run-1"
            / "checkpoint-500",
            PROJECT_ROOT / "outputs" / "online_retail_revenue" / "evaluation",
            PROJECT_ROOT / "outputs" / "online_retail_revenue" / "zero_shot",
        ),
    ]
    for checkpoint, evaluation, zero_shot in candidates:
        metrics_path = evaluation / "finetuned_metrics.csv"
        predictions_path = evaluation / "finetuned_test_predictions.npz"
        if not (checkpoint / "config.json").exists() or not metrics_path.exists() or not predictions_path.exists():
            continue
        metrics = pd.read_csv(metrics_path).iloc[0]
        if all(abs(float(metrics[key]) - expected) <= 1e-6 for key, expected in EXPECTED_FOLD3.items()):
            return checkpoint, evaluation, zero_shot
    return None


def _evaluate_chronos_checkpoint(
    checkpoint: Path,
    context: np.ndarray,
    actual: np.ndarray,
    config: ExperimentConfig,
    device: str,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    samples = predict_samples(
        str(checkpoint), [np.asarray(context, dtype=np.float32)], config, device, batch_size
    )[0]
    metrics = calculate_metrics(
        samples[None, :, :],
        np.asarray(actual, dtype=np.float64)[None, :],
        [np.asarray(context, dtype=np.float64)],
        config.seasonal_period,
    )
    return metrics, samples, _quantiles_from_samples(samples)


def _reuse_fold3(
    spec: FoldSpec,
    fold_dir: Path,
    dates: pd.DatetimeIndex,
) -> bool:
    found = _find_fold3_reuse()
    if found is None:
        return False
    checkpoint, evaluation_dir, zero_shot_dir = found
    metrics = pd.read_csv(evaluation_dir / "finetuned_metrics.csv").iloc[0]
    payload = np.load(evaluation_dir / "finetuned_test_predictions.npz", allow_pickle=False)
    samples = np.asarray(payload["samples"], dtype=np.float64)[0]
    actual = np.asarray(payload["actual"], dtype=np.float64)[0]
    quantiles = _quantiles_from_samples(samples)
    rows = [_metric_row(spec, "chronos_finetuned", "test", {k: float(metrics[k]) for k in ("WQL", "MASE", "MAE", "WAPE")})]
    predictions = _prediction_rows(
        spec,
        "chronos_finetuned",
        "test",
        dates[spec.test_start : spec.test_end],
        actual,
        quantiles,
    )
    validation_source = evaluation_dir / "checkpoint_validation_metrics.csv"
    if validation_source.exists():
        shutil.copy2(validation_source, fold_dir / "chronos_validation_candidates.csv")
    if zero_shot_dir.exists() and (zero_shot_dir / "zero_shot_metrics.csv").exists():
        zero_metrics = pd.read_csv(zero_shot_dir / "zero_shot_metrics.csv").iloc[0]
        zero_payload = np.load(zero_shot_dir / "zero_shot_test_predictions.npz", allow_pickle=False)
        zero_samples = np.asarray(zero_payload["samples"], dtype=np.float64)[0]
        zero_actual = np.asarray(zero_payload["actual"], dtype=np.float64)[0]
        rows.append(
            _metric_row(
                spec,
                "chronos_zero_shot",
                "test",
                {k: float(zero_metrics[k]) for k in ("WQL", "MASE", "MAE", "WAPE")},
            )
        )
        predictions.extend(
            _prediction_rows(
                spec,
                "chronos_zero_shot",
                "test",
                dates[spec.test_start : spec.test_end],
                zero_actual,
                _quantiles_from_samples(zero_samples),
            )
        )
    _write_csv_rows(fold_dir / "chronos_metrics.csv", rows)
    _write_csv_rows(fold_dir / "chronos_predictions.csv", predictions)
    _write_chronos_marker(fold_dir, checkpoint, reused=True)
    return True


def run_chronos_folds(
    data_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    device: str = "cuda",
    batch_size: int = 4,
) -> None:
    dataset = load_online_retail_revenue(data_path)
    target = np.asarray(dataset[0]["target"], dtype=np.float64)
    dates = pd.DatetimeIndex(dataset[0]["timestamp"])
    specs = _load_specs(output_dir)
    for spec in specs:
        fold_dir = output_dir / "folds" / spec.name
        marker = fold_dir / "models" / "chronos" / "selected_checkpoint.txt"
        if marker.exists() and (fold_dir / "chronos_metrics.csv").exists():
            selected = Path(marker.read_text(encoding="utf-8").strip())
            if (selected / "config.json").exists():
                continue
        if spec.fold == len(specs) and _reuse_fold3(spec, fold_dir, dates):
            continue

        arrays = _fold_arrays(target, spec)
        training_root = fold_dir / "chronos" / "training"
        candidates = _checkpoint_candidates(training_root)
        max_step = max(
            [int(path.name.rsplit("-", 1)[-1]) for path in candidates], default=0
        )
        if max_step < config.max_steps:
            resume = find_latest_checkpoint(training_root) if candidates else None
            run_finetuning(
                fold_dir / "arrow" / "train",
                training_root,
                config,
                max_steps=config.max_steps,
                save_steps=config.save_steps,
                batch_size=batch_size,
                resume_from_checkpoint=resume,
                cpu=device == "cpu",
            )
            candidates = _checkpoint_candidates(training_root)
        validation_rows = []
        for checkpoint in candidates:
            metrics, _, _ = _evaluate_chronos_checkpoint(
                checkpoint,
                arrays["validation_context"],
                arrays["validation_truth"],
                config,
                device,
                1,
            )
            validation_rows.append(
                {
                    "fold": spec.fold,
                    "checkpoint": str(checkpoint.resolve()),
                    **metrics,
                }
            )
        validation_frame = pd.DataFrame(validation_rows).sort_values(["WQL", "MASE"])
        validation_frame.to_csv(fold_dir / "chronos_validation_candidates.csv", index=False)
        best_checkpoint = Path(validation_frame.iloc[0]["checkpoint"])
        test_metrics, test_samples, test_quantiles = _evaluate_chronos_checkpoint(
            best_checkpoint,
            arrays["test_context"],
            arrays["test_truth"],
            config,
            device,
            1,
        )
        zero_samples = predict_samples(
            config.model_id,
            [arrays["test_context"].astype(np.float32)],
            config,
            device,
            1,
        )[0]
        zero_quantiles = _quantiles_from_samples(zero_samples)
        zero_metrics = _metrics_from_quantiles(
            zero_quantiles, arrays["test_truth"], arrays["test_context"], config
        )
        metric_rows = [
            _metric_row(spec, "chronos_finetuned", "test", test_metrics),
            _metric_row(spec, "chronos_zero_shot", "test", zero_metrics),
        ]
        prediction_rows = _prediction_rows(
            spec,
            "chronos_finetuned",
            "test",
            dates[spec.test_start : spec.test_end],
            arrays["test_truth"],
            test_quantiles,
        )
        prediction_rows.extend(
            _prediction_rows(
                spec,
                "chronos_zero_shot",
                "test",
                dates[spec.test_start : spec.test_end],
                arrays["test_truth"],
                zero_quantiles,
            )
        )
        _write_csv_rows(fold_dir / "chronos_metrics.csv", metric_rows)
        _write_csv_rows(fold_dir / "chronos_predictions.csv", prediction_rows)
        np.savez_compressed(
            fold_dir / "chronos_test_predictions.npz",
            samples=test_samples[None, :, :].astype(np.float32),
            actual=arrays["test_truth"][None, :].astype(np.float32),
        )
        _write_chronos_marker(fold_dir, best_checkpoint, reused=False)


def _tree_artifact_paths(models_dir: Path, model_name: str) -> list[Path]:
    if model_name == "random_forest":
        return [models_dir / "random_forest.joblib"]
    if model_name == "xgboost":
        return [models_dir / "xgboost.json", models_dir / "xgboost_residuals.npy"]
    return [models_dir / "catboost.cbm", models_dir / "catboost_residuals.npy"]


def _save_tree_model(
    models_dir: Path, model_name: str, model, residuals: np.ndarray, params: dict
) -> tuple[list[Path], Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = _tree_artifact_paths(models_dir, model_name)
    if model_name == "random_forest":
        joblib.dump({"model": model, "residuals": residuals, "params": params}, paths[0])
        loaded = joblib.load(paths[0])["model"]
    elif model_name == "xgboost":
        model.save_model(paths[0])
        np.save(paths[1], residuals)
        loaded = _make_model("xgboost", 42, params)
        loaded.load_model(paths[0])
    else:
        model.save_model(paths[0])
        np.save(paths[1], residuals)
        from catboost import CatBoostRegressor

        loaded = CatBoostRegressor()
        loaded.load_model(paths[0])
    return paths, loaded


def _stat_quantiles(result: dict) -> np.ndarray:
    keys = ["lo-80", "lo-60", "lo-40", "lo-20", "mean", "hi-20", "hi-40", "hi-60", "hi-80"]
    missing = [key for key in keys if key not in result]
    if missing:
        raise ValueError(f"Statistical forecast is missing quantiles: {missing}")
    return np.stack([np.asarray(result[key], dtype=np.float64) for key in keys])


def _model_manifest(
    spec: FoldSpec,
    selected: dict[str, dict],
    artifact_paths: dict[str, list[Path]],
) -> dict:
    artifacts = {}
    for model, paths in artifact_paths.items():
        artifacts[model] = [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
            if isinstance(path, Path) and path.is_file()
        ]
    return {
        "fold": asdict(spec),
        "seed": 42,
        "features": {"lags": list(LAGS), "rolling_windows": list(ROLLING_WINDOWS)},
        "selected_settings": selected,
        "artifacts": artifacts,
        "versions": _versions(),
    }


def run_baseline_folds(
    data_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    try:
        from statsforecast.models import AutoARIMA, AutoETS
    except ImportError as error:
        raise ImportError(
            "StatsForecast is required. Install project extras with: pip install -e '.[comparison-models]'"
        ) from error

    dataset = load_online_retail_revenue(data_path)
    target = np.asarray(dataset[0]["target"], dtype=np.float64)
    dates = pd.DatetimeIndex(dataset[0]["timestamp"])
    specs = _load_specs(output_dir)
    for spec in specs:
        fold_dir = output_dir / "folds" / spec.name
        completion = fold_dir / "baselines_complete.json"
        if completion.exists():
            recorded = json.loads(completion.read_text(encoding="utf-8"))
            if all(Path(item["path"]).is_file() and _sha256(Path(item["path"])) == item["sha256"] for item in recorded["artifacts"]):
                continue
        arrays = _fold_arrays(target, spec)
        models_dir = fold_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        metric_rows = []
        prediction_rows = []
        candidate_rows = []
        selected_settings: dict[str, dict] = {}
        artifact_paths: dict[str, list[Path]] = {}

        naive_samples = seasonal_naive_samples(
            [arrays["test_context"]], config.prediction_length, config.num_samples, period=7
        )[0]
        naive_quantiles = _quantiles_from_samples(naive_samples)
        naive_metrics = _metrics_from_quantiles(
            naive_quantiles, arrays["test_truth"], arrays["test_context"], config
        )
        metric_rows.append(_metric_row(spec, "seasonal_naive_7", "test", naive_metrics))
        prediction_rows.extend(
            _prediction_rows(
                spec,
                "seasonal_naive_7",
                "test",
                dates[spec.test_start : spec.test_end],
                arrays["test_truth"],
                naive_quantiles,
            )
        )

        for model_index, (model_name, candidates) in enumerate(TREE_CANDIDATES.items()):
            fitted_candidates = []
            for candidate_index, params in enumerate(candidates):
                model, residuals, audit = _fit_model_and_residuals(
                    model_name,
                    arrays["train"],
                    dates[: spec.train_end],
                    config.seed,
                    params=params,
                )
                samples = recursive_samples(
                    model,
                    arrays["validation_context"],
                    dates[spec.validation_start : spec.validation_end],
                    residuals,
                    config.num_samples,
                    config.seed + model_index * 100 + candidate_index,
                )
                quantiles = _quantiles_from_samples(samples)
                metrics = _metrics_from_quantiles(
                    quantiles,
                    arrays["validation_truth"],
                    arrays["validation_context"],
                    config,
                )
                candidate_rows.append(
                    {
                        "fold": spec.fold,
                        "model": model_name,
                        "candidate": candidate_index + 1,
                        "params": json.dumps(params, sort_keys=True),
                        **metrics,
                    }
                )
                fitted_candidates.append((metrics, params, model, residuals, audit))
            selected = min(fitted_candidates, key=lambda item: (item[0]["WAPE"], item[0]["MASE"]))
            _, params, model, residuals, audit = selected
            test_samples = recursive_samples(
                model,
                arrays["test_context"],
                dates[spec.test_start : spec.test_end],
                residuals,
                config.num_samples,
                config.seed + model_index * 1000,
            )
            test_quantiles = _quantiles_from_samples(test_samples)
            test_metrics = _metrics_from_quantiles(
                test_quantiles, arrays["test_truth"], arrays["test_context"], config
            )
            metric_rows.append(_metric_row(spec, model_name, "test", test_metrics))
            prediction_rows.extend(
                _prediction_rows(
                    spec,
                    model_name,
                    "test",
                    dates[spec.test_start : spec.test_end],
                    arrays["test_truth"],
                    test_quantiles,
                )
            )
            saved_paths, loaded_model = _save_tree_model(
                models_dir, model_name, model, residuals, params
            )
            probe = pd.DataFrame([_feature_row(arrays["test_context"], dates[spec.test_start])])
            if not np.allclose(model.predict(probe), loaded_model.predict(probe), rtol=1e-7, atol=1e-6):
                raise AssertionError(f"Reloaded {model_name} does not reproduce predictions.")
            artifact_paths[model_name] = saved_paths
            selected_settings[model_name] = {"params": params, "audit": audit}

        stat_models = {
            "auto_arima": AutoARIMA(season_length=7),
            "auto_ets": AutoETS(season_length=7, model="ZZZ"),
        }
        for model_name, model in stat_models.items():
            model.fit(arrays["train"])
            validation_result = model.predict(config.prediction_length, level=STAT_LEVELS)
            validation_quantiles = _stat_quantiles(validation_result)
            validation_metrics = _metrics_from_quantiles(
                validation_quantiles,
                arrays["validation_truth"],
                arrays["validation_context"],
                config,
            )
            candidate_rows.append(
                {
                    "fold": spec.fold,
                    "model": model_name,
                    "candidate": 1,
                    "params": json.dumps({"season_length": 7}),
                    **validation_metrics,
                }
            )
            test_result = model.forward(
                arrays["test_context"], config.prediction_length, level=STAT_LEVELS
            )
            test_quantiles = _stat_quantiles(test_result)
            test_metrics = _metrics_from_quantiles(
                test_quantiles, arrays["test_truth"], arrays["test_context"], config
            )
            metric_rows.append(_metric_row(spec, model_name, "test", test_metrics))
            prediction_rows.extend(
                _prediction_rows(
                    spec,
                    model_name,
                    "test",
                    dates[spec.test_start : spec.test_end],
                    arrays["test_truth"],
                    test_quantiles,
                )
            )
            model_path = models_dir / f"{model_name}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(model, handle)
            with model_path.open("rb") as handle:
                loaded = pickle.load(handle)
            reloaded = _stat_quantiles(
                loaded.forward(arrays["test_context"], config.prediction_length, level=STAT_LEVELS)
            )
            if not np.allclose(test_quantiles, reloaded, rtol=1e-7, atol=1e-6):
                raise AssertionError(f"Reloaded {model_name} does not reproduce predictions.")
            artifact_paths[model_name] = [model_path]
            selected_settings[model_name] = {"season_length": 7, "selection": "automatic"}

        _write_csv_rows(fold_dir / "baseline_metrics.csv", metric_rows)
        _write_csv_rows(fold_dir / "baseline_predictions.csv", prediction_rows)
        _write_csv_rows(fold_dir / "validation_candidates.csv", candidate_rows)
        manifest = _model_manifest(spec, selected_settings, artifact_paths)
        chronos_selection = models_dir / "chronos" / "selection.json"
        if chronos_selection.exists():
            recorded_chronos = json.loads(chronos_selection.read_text(encoding="utf-8"))
            manifest["selected_settings"]["chronos_finetuned"] = {
                "checkpoint": recorded_chronos["checkpoint"],
                "selection_metric": "validation_WQL",
                "reused_existing_fold3": recorded_chronos["reused_existing_fold3"],
            }
            manifest["artifacts"]["chronos_finetuned"] = recorded_chronos["artifacts"]
        manifest_path = models_dir / "model_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        recorded_artifacts = [item for items in manifest["artifacts"].values() for item in items]
        completion.write_text(
            json.dumps({"artifacts": recorded_artifacts}, indent=2), encoding="utf-8"
        )


def summarize_rolling_benchmark(output_dir: Path) -> pd.DataFrame:
    specs = _load_specs(output_dir)
    metric_frames = []
    prediction_frames = []
    candidate_frames = []
    for spec in specs:
        fold_dir = output_dir / "folds" / spec.name
        for filename in ("chronos_metrics.csv", "baseline_metrics.csv"):
            path = fold_dir / filename
            if path.exists():
                metric_frames.append(pd.read_csv(path))
        for filename in ("chronos_predictions.csv", "baseline_predictions.csv"):
            path = fold_dir / filename
            if path.exists():
                prediction_frames.append(pd.read_csv(path))
        path = fold_dir / "validation_candidates.csv"
        if path.exists():
            candidate_frames.append(pd.read_csv(path))
        chronos_path = fold_dir / "chronos_validation_candidates.csv"
        if chronos_path.exists():
            chronos_candidates = pd.read_csv(chronos_path)
            if "checkpoint" in chronos_candidates:
                chronos_candidates["params"] = chronos_candidates["checkpoint"]
            elif "model" in chronos_candidates:
                chronos_candidates["params"] = chronos_candidates["model"]
            chronos_candidates["model"] = "chronos_finetuned"
            candidate_frames.append(chronos_candidates)
    if not metric_frames:
        raise FileNotFoundError("No fold metrics were found.")
    metrics = pd.concat(metric_frames, ignore_index=True).drop_duplicates(
        ["fold", "model", "split"], keep="last"
    )
    required = {(fold, model) for fold in range(1, len(specs) + 1) for model in RANKED_MODELS}
    observed = set(
        metrics.loc[metrics["model"].isin(RANKED_MODELS), ["fold", "model"]]
        .itertuples(index=False, name=None)
    )
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"Cannot summarize; missing fold/model results: {missing}")
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).drop_duplicates(
            ["fold", "model", "split", "date"], keep="last"
        ).to_csv(output_dir / "predictions.csv", index=False)
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            output_dir / "validation_candidates.csv", index=False
        )

    ranked = metrics.loc[metrics["model"].isin(RANKED_MODELS)].copy()
    summary_rows = []
    for model, frame in ranked.groupby("model"):
        row = {"model": model, "folds": len(frame)}
        for metric in ("WQL", "MASE", "MAE", "WAPE"):
            row[f"mean_{metric}"] = float(frame[metric].mean())
            row[f"std_{metric}"] = float(frame[metric].std(ddof=0))
            row[f"worst_{metric}"] = float(frame[metric].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    ranking = summary.sort_values(["mean_WAPE", "mean_MASE", "mean_MAE"]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["chronos_top_3"] = (ranking["model"] == "chronos_finetuned") & (ranking["rank"] <= 3)
    ranking.to_csv(output_dir / "model_ranking.csv", index=False)

    chronos_rank = int(ranking.loc[ranking["model"] == "chronos_finetuned", "rank"].iloc[0])
    decision = {
        "criterion": "Fine-tuned Chronos ranks in the top 3 by mean test WAPE; ties use mean MASE then mean MAE.",
        "chronos_rank": chronos_rank,
        "pass": chronos_rank <= 3,
        "ranked_models": list(RANKED_MODELS),
    }
    (output_dir / "validity_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    plot_frame = ranking.sort_values("mean_WAPE")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(plot_frame))
    axis.bar(
        positions,
        plot_frame["mean_WAPE"] * 100,
        yerr=plot_frame["std_WAPE"] * 100,
        color=["#0f766e" if model == "chronos_finetuned" else "#64748b" for model in plot_frame["model"]],
        capsize=4,
    )
    axis.set_xticks(positions, plot_frame["model"], rotation=25, ha="right")
    axis.set_ylabel("Mean WAPE (%)")
    axis.set_title("Three-fold daily revenue forecast comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(figure)
    return ranking


def run_rolling_benchmark(
    stage: str,
    data_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    folds: int = 3,
    device: str = "cuda",
    batch_size: int = 4,
) -> pd.DataFrame | None:
    allowed = {"prepare", "chronos", "baselines", "summarize", "all"}
    if stage not in allowed:
        raise ValueError(f"stage must be one of {sorted(allowed)}")
    if stage in {"prepare", "all"}:
        prepare_rolling_benchmark(data_path, output_dir, config, folds)
    if stage in {"chronos", "all"}:
        run_chronos_folds(data_path, output_dir, config, device, batch_size)
    if stage in {"baselines", "all"}:
        run_baseline_folds(data_path, output_dir, config)
    if stage in {"summarize", "all"}:
        return summarize_rolling_benchmark(output_dir)
    return None
