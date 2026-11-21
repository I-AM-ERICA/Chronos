from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import evaluation_arrays
from .metrics import calculate_metrics


EvaluationSplit = Literal["validation", "test"]


def seasonal_naive_samples(
    contexts: list[np.ndarray], prediction_length: int, num_samples: int, period: int = 7
) -> np.ndarray:
    forecasts = []
    for context in contexts:
        history = np.asarray(context, dtype=np.float32)
        if len(history) < period:
            raise ValueError(f"At least {period} observations are required.")
        forecast = np.resize(history[-period:], prediction_length)
        forecasts.append(np.repeat(forecast[None, :], num_samples, axis=0))
    return np.stack(forecasts)


def prediction_dates(
    row: dict, config: ExperimentConfig, split_name: EvaluationSplit
) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(row["timestamp"])
    if split_name == "validation":
        return timestamps[-config.holdout_length : -config.prediction_length]
    if split_name == "test":
        return timestamps[-config.prediction_length :]
    raise ValueError("split_name must be validation or test")


def write_revenue_forecasts(
    prediction_path: Path,
    dataset: list[dict],
    config: ExperimentConfig,
    output_dir: Path,
    label: str,
    split_name: EvaluationSplit = "test",
    entity_column: str = "store_id",
    entity_label: str = "store",
) -> dict[str, float | int | str]:
    payload = np.load(prediction_path, allow_pickle=False)
    samples = np.asarray(payload["samples"], dtype=np.float64)
    actual = np.asarray(payload["actual"], dtype=np.float64)
    series_ids = payload["series_ids"].astype(str)
    row_by_id = {str(row["id"]): row for row in dataset}
    if set(series_ids) - set(row_by_id):
        raise ValueError("Prediction IDs do not match the prepared revenue dataset.")

    quantiles = {
        "pred_q10": np.quantile(samples, 0.1, axis=1),
        "pred_q50": np.quantile(samples, 0.5, axis=1),
        "pred_q90": np.quantile(samples, 0.9, axis=1),
    }
    forecast_rows = []
    per_store_rows = []
    contexts, expected_actual, expected_ids = evaluation_arrays(
        dataset, config, split_name=split_name
    )
    expected_context = {series_id: context for series_id, context in zip(expected_ids, contexts)}
    expected_truth = {series_id: truth for series_id, truth in zip(expected_ids, expected_actual)}
    for series_index, store_id in enumerate(series_ids):
        dates = prediction_dates(row_by_id[store_id], config, split_name)
        if len(dates) != samples.shape[2]:
            raise ValueError("Prediction horizon does not match revenue dates.")
        if not np.allclose(actual[series_index], expected_truth[store_id]):
            raise ValueError(f"Saved actual values do not match store {store_id}.")
        for day_index, date in enumerate(dates):
            forecast_rows.append(
                {
                    entity_column: store_id,
                    "date": date.date().isoformat(),
                    "actual_revenue": actual[series_index, day_index],
                    "pred_q10": quantiles["pred_q10"][series_index, day_index],
                    "pred_q50": quantiles["pred_q50"][series_index, day_index],
                    "pred_q90": quantiles["pred_q90"][series_index, day_index],
                }
            )
        store_metrics = calculate_metrics(
            samples[series_index : series_index + 1],
            actual[series_index : series_index + 1],
            [expected_context[store_id]],
            config.seasonal_period,
        )
        per_store_rows.append(
            {"label": label, "split": split_name, entity_column: store_id, **store_metrics}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecast_rows).to_csv(
        output_dir / f"{label}_{split_name}_{entity_label}_forecasts.csv", index=False
    )
    pd.DataFrame(per_store_rows).to_csv(
        output_dir / f"{label}_{split_name}_{entity_label}_metrics.csv", index=False
    )

    rng = np.random.default_rng(config.seed)
    store_count, sample_count, horizon = samples.shape
    total_draws = np.zeros((config.monte_carlo_draws, horizon), dtype=np.float64)
    for store_index in range(store_count):
        chosen = rng.integers(0, sample_count, size=config.monte_carlo_draws)
        total_draws += samples[store_index, chosen, :]
    total_actual = actual.sum(axis=0)
    first_dates = prediction_dates(row_by_id[series_ids[0]], config, split_name)
    total_frame = pd.DataFrame(
        {
            "date": [date.date().isoformat() for date in first_dates],
            "actual_total_revenue": total_actual,
            "pred_total_q10": np.quantile(total_draws, 0.1, axis=0),
            "pred_total_q50": np.quantile(total_draws, 0.5, axis=0),
            "pred_total_q90": np.quantile(total_draws, 0.9, axis=0),
        }
    )
    total_frame.to_csv(
        output_dir / f"{label}_{split_name}_total_forecast.csv", index=False
    )
    draw_period_totals = total_draws.sum(axis=1)
    actual_period_total = float(total_actual.sum())
    predicted_period_median = float(np.quantile(draw_period_totals, 0.5))
    summary: dict[str, float | int | str] = {
        "label": label,
        "split": split_name,
        f"{entity_label}_count": store_count,
        "actual_28d_total": actual_period_total,
        "pred_28d_q10": float(np.quantile(draw_period_totals, 0.1)),
        "pred_28d_q50": predicted_period_median,
        "pred_28d_q90": float(np.quantile(draw_period_totals, 0.9)),
        "total_absolute_error": abs(actual_period_total - predicted_period_median),
        "total_ape": (
            abs(actual_period_total - predicted_period_median) / actual_period_total
            if actual_period_total > 0
            else float("nan")
        ),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / f"{label}_{split_name}_total_summary.csv", index=False
    )
    return summary


def save_naive_predictions(
    dataset: list[dict],
    config: ExperimentConfig,
    output_dir: Path,
    split_name: EvaluationSplit = "test",
    label: str = "seasonal_naive_7",
    dataset_name: str = "online-retail-revenue",
) -> tuple[dict[str, float | int | str], Path]:
    contexts, actual, series_ids = evaluation_arrays(dataset, config, split_name)
    samples = seasonal_naive_samples(
        contexts, config.prediction_length, config.num_samples, period=7
    )
    metrics = calculate_metrics(samples, actual, contexts, config.seasonal_period)
    row: dict[str, float | int | str] = {
        "dataset": dataset_name,
        "split": split_name,
        "model": "seasonal-naive-7",
        "label": label,
        "series_count": len(dataset),
        **metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_{split_name}_predictions.npz"
    np.savez_compressed(
        path,
        samples=samples.astype(np.float32),
        actual=actual.astype(np.float32),
        series_ids=np.asarray(series_ids),
    )
    return row, path
