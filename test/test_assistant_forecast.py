from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from chronos_sales.assistant_forecast import (
    FORECAST_DAYS,
    NUM_SAMPLES,
    forecast_with_pipeline,
    load_chronos_pipeline,
    select_forecast,
)
from chronos_sales.assistant_storage import RevenueRecord
from chronos_sales.config import PROJECT_ROOT


def _records(days: int = 60) -> list[RevenueRecord]:
    start = date(2026, 1, 1)
    return [
        RevenueRecord(
            business_id="default",
            date=start + timedelta(days=index),
            revenue=10_000 + index,
            source="test",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        for index in range(days)
    ]


class MockPipeline:
    def predict(self, contexts, prediction_length, num_samples):
        assert len(contexts[0]) == 28
        assert prediction_length == FORECAST_DAYS
        assert num_samples == NUM_SAMPLES
        values = np.tile(np.arange(1, FORECAST_DAYS + 1), (NUM_SAMPLES, 1))
        values[0, 0] = -999
        return torch.tensor(values[None, :, :], dtype=torch.float32)


def test_mock_chronos_generates_exact_next_14_dates_and_quantiles():
    records = _records()
    points = forecast_with_pipeline(MockPipeline(), records)
    assert len(points) == 14
    assert points[0].base_date == records[-1].date
    assert points[0].forecast_date == records[-1].date + timedelta(days=1)
    assert points[-1].forecast_date == records[-1].date + timedelta(days=14)
    assert points[7].days_ahead == 8
    assert points[0].pred_q10 == 1
    assert points[0].pred_q50 == 1
    assert points[-1].pred_q90 == 14
    assert select_forecast(points, points[7].forecast_date) == points[7]


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CHRONOS_INTEGRATION") != "1",
    reason="set RUN_CHRONOS_INTEGRATION=1 to load the real checkpoint",
)
def test_real_release_checkpoint_cpu_inference():
    model_path = Path(
        os.getenv(
            "CHRONOS_MODEL_PATH",
            str(
                PROJECT_ROOT
                / "outputs"
                / "release"
                / "chronos-online-retail-v0.1"
                / "inference"
            ),
        )
    )
    assert (model_path / "model.safetensors").exists()
    pipeline = load_chronos_pipeline(model_path, device="cpu")
    points = forecast_with_pipeline(pipeline, _records())
    assert len(points) == 14
    assert all(point.pred_q10 >= 0 for point in points)
