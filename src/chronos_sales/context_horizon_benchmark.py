from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .assistant_forecast import load_chronos_pipeline, model_identifier
from .metrics import calculate_metrics


SMOKE_GRID = {7: (14, 28), 14: (28, 56), 28: (56, 60)}
FULL_GRID = {7: (14, 28, 56), 14: (28, 56, 112), 28: (56, 60, 112, 224)}
KEY_COLUMNS = ["horizon_days", "context_days", "fold"]


@dataclass(frozen=True)
class BenchmarkTask:
    horizon_days: int
    context_days: int
    fold: int
    cutoff: int
    context_start: int
    forecast_end: int


def load_daily_revenue(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"date", "net_revenue_gbp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Daily revenue data is missing columns: {sorted(missing)}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    values = frame["net_revenue_gbp"].to_numpy(dtype=np.float32)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Daily revenue dates must be unique and sorted.")
    expected = pd.date_range(dates[0], dates[-1], freq="D")
    if not dates.equals(expected):
        raise ValueError("Daily revenue dates must be consecutive.")
    if not np.isfinite(values).all():
        raise ValueError("Daily revenue contains non-finite values.")
    return dates, values


def build_tasks(
    series_length: int,
    *,
    smoke: bool,
    folds: int,
    fold_step: int = 28,
) -> list[BenchmarkTask]:
    grid = SMOKE_GRID if smoke else FULL_GRID
    max_horizon = max(grid)
    tasks: list[BenchmarkTask] = []
    for fold_index in range(folds):
        cutoff = series_length - max_horizon - fold_index * fold_step
        for horizon, contexts in grid.items():
            for context in contexts:
                context_start = cutoff - context
                forecast_end = cutoff + horizon
                if context_start < 0 or forecast_end > series_length:
                    raise ValueError(
                        f"Series length {series_length} is too short for fold {fold_index + 1}, "
                        f"context {context}, horizon {horizon}."
                    )
                tasks.append(
                    BenchmarkTask(
                        horizon_days=horizon,
                        context_days=context,
                        fold=fold_index + 1,
                        cutoff=cutoff,
                        context_start=context_start,
                        forecast_end=forecast_end,
                    )
                )
    return sorted(tasks, key=lambda task: (task.horizon_days, task.context_days, task.fold))


def _prediction_rows(
    task: BenchmarkTask,
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    samples: np.ndarray,
) -> list[dict[str, object]]:
    q10, q50, q90 = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    forecast_dates = dates[task.cutoff : task.forecast_end]
    return [
        {
            "horizon_days": task.horizon_days,
            "context_days": task.context_days,
            "fold": task.fold,
            "date": day.date().isoformat(),
            "actual_revenue": float(actual[index]),
            "pred_q10": float(q10[index]),
            "pred_q50": float(q50[index]),
            "pred_q90": float(q90[index]),
        }
        for index, day in enumerate(forecast_dates)
    ]


def _metric_row(
    task: BenchmarkTask,
    dates: pd.DatetimeIndex,
    context: np.ndarray,
    actual: np.ndarray,
    samples: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, object]:
    metrics = calculate_metrics(
        samples[None, :, :], actual[None, :], [context], seasonal_period=7
    )
    q10, q90 = np.quantile(samples, [0.1, 0.9], axis=0)
    coverage = float(np.mean((actual >= q10) & (actual <= q90)))
    return {
        "horizon_days": task.horizon_days,
        "context_days": task.context_days,
        "fold": task.fold,
        "context_start": dates[task.context_start].date().isoformat(),
        "context_end": dates[task.cutoff - 1].date().isoformat(),
        "forecast_start": dates[task.cutoff].date().isoformat(),
        "forecast_end": dates[task.forecast_end - 1].date().isoformat(),
        **metrics,
        "interval_coverage_q10_q90": coverage,
        "inference_seconds": elapsed_seconds,
    }


def _write_progress(
    output_dir: Path,
    metrics: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> None:
    metric_frame = pd.DataFrame(metrics).drop_duplicates(KEY_COLUMNS, keep="last")
    metric_frame = metric_frame.sort_values(KEY_COLUMNS)
    metric_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    prediction_frame = pd.DataFrame(predictions).drop_duplicates(
        KEY_COLUMNS + ["date"], keep="last"
    )
    prediction_frame = prediction_frame.sort_values(KEY_COLUMNS + ["date"])
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)


def _summarize(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    summary = (
        metrics.groupby(["horizon_days", "context_days"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_WAPE=("WAPE", "mean"),
            std_WAPE=("WAPE", "std"),
            worst_WAPE=("WAPE", "max"),
            mean_MASE=("MASE", "mean"),
            mean_MAE=("MAE", "mean"),
            mean_WQL=("WQL", "mean"),
            mean_interval_coverage=("interval_coverage_q10_q90", "mean"),
            mean_inference_seconds=("inference_seconds", "mean"),
        )
        .sort_values(["horizon_days", "context_days"])
    )
    summary["std_WAPE"] = summary["std_WAPE"].fillna(0.0)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    recommendation_rows: list[dict[str, object]] = []
    for horizon, group in summary.groupby("horizon_days"):
        best_wape = float(group["mean_WAPE"].min())
        eligible = group[group["mean_WAPE"] <= best_wape + 0.02]
        chosen = eligible.sort_values(["context_days", "mean_WAPE"]).iloc[0]
        recommendation_rows.append(
            {
                "horizon_days": int(horizon),
                "recommended_context_days": int(chosen["context_days"]),
                "recommended_mean_WAPE": float(chosen["mean_WAPE"]),
                "best_mean_WAPE": best_wape,
                "selection_rule": "shortest context within 2 percentage points of best mean WAPE",
            }
        )
    recommendations = pd.DataFrame(recommendation_rows)
    recommendations.to_csv(output_dir / "recommendations.csv", index=False)
    _plot_wape_heatmap(summary, output_dir / "wape_heatmap.png")
    return summary, recommendations


def _plot_wape_heatmap(summary: pd.DataFrame, path: Path) -> None:
    horizons = sorted(summary["horizon_days"].unique())
    contexts = sorted(summary["context_days"].unique())
    matrix = np.full((len(horizons), len(contexts)), np.nan, dtype=np.float64)
    for _, row in summary.iterrows():
        y = horizons.index(int(row["horizon_days"]))
        x = contexts.index(int(row["context_days"]))
        matrix[y, x] = float(row["mean_WAPE"])
    fig, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(matrix, cmap="YlGnBu_r", aspect="auto", vmin=0)
    axis.set_xticks(range(len(contexts)), [str(value) for value in contexts])
    axis.set_yticks(range(len(horizons)), [str(value) for value in horizons])
    axis.set_xlabel("Context days")
    axis.set_ylabel("Forecast horizon days")
    axis.set_title("Chronos mean WAPE by context and horizon")
    for y in range(len(horizons)):
        for x in range(len(contexts)):
            if np.isfinite(matrix[y, x]):
                axis.text(x, y, f"{matrix[y, x]:.1%}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Mean WAPE")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_context_horizon_benchmark(
    *,
    data_path: Path,
    model_path: Path,
    output_dir: Path,
    smoke: bool = False,
    folds: int | None = None,
    device: str = "cpu",
    num_samples: int = 20,
    seed: int = 42,
    pipeline: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = folds or (2 if smoke else 10)
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    dates, values = load_daily_revenue(data_path)
    tasks = build_tasks(len(values), smoke=smoke, folds=folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "smoke": smoke,
        "folds": folds,
        "fold_step_days": 28,
        "grid": {str(key): list(value) for key, value in (SMOKE_GRID if smoke else FULL_GRID).items()},
        "num_samples": num_samples,
        "seed": seed,
        "device": device,
        "data_path": str(data_path.resolve()),
        "data_start": dates[0].date().isoformat(),
        "data_end": dates[-1].date().isoformat(),
        "model_path": str(model_path.resolve()),
        "model_identifier": model_identifier(model_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                "Output directory contains a different benchmark configuration. "
                "Choose another output directory."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    metric_path = output_dir / "fold_metrics.csv"
    prediction_path = output_dir / "predictions.csv"
    metrics: list[dict[str, object]] = (
        pd.read_csv(metric_path).to_dict("records") if metric_path.exists() else []
    )
    predictions: list[dict[str, object]] = (
        pd.read_csv(prediction_path).to_dict("records")
        if prediction_path.exists()
        else []
    )
    completed = {
        (
            int(str(row["horizon_days"])),
            int(str(row["context_days"])),
            int(str(row["fold"])),
        )
        for row in metrics
    }
    remaining = [
        task
        for task in tasks
        if (task.horizon_days, task.context_days, task.fold) not in completed
    ]
    if remaining and pipeline is None:
        pipeline = load_chronos_pipeline(model_path, device=device)

    for task_index, task in enumerate(remaining, start=1):
        assert pipeline is not None
        context = values[task.context_start : task.cutoff].copy()
        actual = values[task.cutoff : task.forecast_end].copy()
        task_seed = seed + task.horizon_days * 10_000 + task.context_days * 10 + task.fold
        torch.manual_seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)
        started = time.perf_counter()
        raw = pipeline.predict(
            [torch.tensor(context, dtype=torch.float32)],
            prediction_length=task.horizon_days,
            num_samples=num_samples,
        )
        elapsed = time.perf_counter() - started
        samples = raw.detach().cpu().float().numpy()[0]
        if samples.shape != (num_samples, task.horizon_days):
            raise RuntimeError(f"Unexpected Chronos output shape: {samples.shape}")
        metrics.append(
            _metric_row(task, dates, context, actual, samples, elapsed)
        )
        predictions.extend(_prediction_rows(task, dates, actual, samples))
        _write_progress(output_dir, metrics, predictions)
        print(
            f"[{task_index}/{len(remaining)}] horizon={task.horizon_days} "
            f"context={task.context_days} fold={task.fold} elapsed={elapsed:.1f}s",
            flush=True,
        )

    return _summarize(output_dir)
