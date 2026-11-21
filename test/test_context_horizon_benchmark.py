from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from chronos_sales.context_horizon_benchmark import (
    build_tasks,
    run_context_horizon_benchmark,
)


class BenchmarkMockPipeline:
    def __init__(self):
        self.calls = 0

    def predict(self, contexts, prediction_length, num_samples):
        self.calls += 1
        last_value = float(contexts[0][-1])
        values = np.full((1, num_samples, prediction_length), last_value)
        return torch.tensor(values, dtype=torch.float32)


def _write_daily_revenue(path, days=180):
    start = date(2025, 1, 1)
    frame = pd.DataFrame(
        {
            "date": [start + timedelta(days=index) for index in range(days)],
            "net_revenue_gbp": [
                10_000 + 1_000 * (index % 7) + index * 10 for index in range(days)
            ],
        }
    )
    frame.to_csv(path, index=False)


def test_smoke_tasks_use_same_cutoff_without_future_leakage():
    tasks = build_tasks(180, smoke=True, folds=2)
    assert len(tasks) == 12
    assert {task.cutoff for task in tasks if task.fold == 1} == {152}
    assert {task.cutoff for task in tasks if task.fold == 2} == {124}
    assert all(task.context_start < task.cutoff < task.forecast_end for task in tasks)


def test_smoke_benchmark_saves_metrics_predictions_plot_and_resumes(tmp_path):
    data_path = tmp_path / "daily_revenue.csv"
    model_path = tmp_path / "model"
    output_dir = tmp_path / "benchmark"
    model_path.mkdir()
    _write_daily_revenue(data_path)
    pipeline = BenchmarkMockPipeline()

    summary, recommendations = run_context_horizon_benchmark(
        data_path=data_path,
        model_path=model_path,
        output_dir=output_dir,
        smoke=True,
        folds=2,
        num_samples=5,
        pipeline=pipeline,
    )
    assert pipeline.calls == 12
    assert len(pd.read_csv(output_dir / "fold_metrics.csv")) == 12
    assert len(pd.read_csv(output_dir / "predictions.csv")) == 196
    assert len(summary) == 6
    assert len(recommendations) == 3
    assert (output_dir / "summary_metrics.csv").exists()
    assert (output_dir / "recommendations.csv").exists()
    assert (output_dir / "wape_heatmap.png").stat().st_size > 0
    assert (output_dir / "run_manifest.json").exists()

    resume_pipeline = BenchmarkMockPipeline()
    resumed_summary, _ = run_context_horizon_benchmark(
        data_path=data_path,
        model_path=model_path,
        output_dir=output_dir,
        smoke=True,
        folds=2,
        num_samples=5,
        pipeline=resume_pipeline,
    )
    assert resume_pipeline.calls == 0
    pd.testing.assert_frame_equal(summary, resumed_summary)
