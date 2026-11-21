from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import torch

from chronos_sales.assistant_service import RevenueAssistantService


class ServiceMockPipeline:
    def predict(self, contexts, prediction_length, num_samples):
        values = np.tile(np.arange(1, prediction_length + 1), (num_samples, 1))
        return torch.tensor(values[None, :, :], dtype=torch.float32)


def test_service_is_ui_independent_and_returns_api_ready_forecast(tmp_path):
    service = RevenueAssistantService(tmp_path / "revenue.sqlite3", tmp_path / "model")
    start = date(2026, 1, 1)
    csv_text = "date,revenue\n" + "\n".join(
        f"{(start + timedelta(days=index)).isoformat()},{100_000 + index}"
        for index in range(60)
    )
    service.import_csv(csv_text)
    run = service.run_forecast(ServiceMockPipeline())
    assert len(run.points) == 14
    point = service.select(run.points, run.points[7].forecast_date)
    assert point.as_api_dict()["days_ahead"] == 8
    answer = service.answer(point, None)
    assert answer.source == "template"
    assert "8일 뒤" in answer.message
