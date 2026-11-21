from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import torch
from fastapi.testclient import TestClient

from chronos_sales.assistant_api import create_app


class ApiMockPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, contexts, prediction_length, num_samples):
        self.calls += 1
        assert len(contexts[0]) == 28
        assert prediction_length == 14
        values = np.tile(
            np.arange(500_000, 500_000 + prediction_length),
            (num_samples, 1),
        )
        return torch.tensor(values[None, :, :], dtype=torch.float32)


def _csv(days: int) -> bytes:
    start = date(2026, 1, 1)
    rows = ["date,revenue"]
    rows.extend(
        f"{(start + timedelta(days=index)).isoformat()},{100_000 + index}"
        for index in range(days)
    )
    return "\n".join(rows).encode()


def test_api_import_forecast_date_answer_and_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOVA_STUDIO_API_KEY", raising=False)
    model_path = tmp_path / "model"
    model_path.mkdir()
    pipeline = ApiMockPipeline()
    app = create_app(
        db_path=tmp_path / "revenue.sqlite3",
        model_path=model_path,
        pipeline_override=pipeline,
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["forecast_days"] == 14

        imported = client.post(
            "/v1/businesses/demo-store/revenue/import",
            files={"file": ("revenue.csv", _csv(28), "text/csv")},
        )
        assert imported.status_code == 200
        assert imported.json()["inserted"] == 28

        forecast = client.post("/v1/businesses/demo-store/forecast")
        assert forecast.status_code == 200
        assert len(forecast.json()["points"]) == 14
        assert forecast.json()["context_days"] == 28

        repeated = client.post("/v1/businesses/demo-store/forecast")
        assert repeated.status_code == 200
        assert pipeline.calls == 1

        selected = client.get(
            "/v1/businesses/demo-store/forecast/2026-01-29"
        )
        assert selected.status_code == 200
        assert selected.json()["forecast"]["pred_q50"] == 500_000
        assert selected.json()["message"] == (
            "1일 뒤 1월 29일 예상 매출액은 500,000원입니다."
        )
        assert selected.json()["message_source"] == "template"

        updated = client.put(
            "/v1/businesses/demo-store/revenue/2026-01-28",
            json={"revenue": 777_000},
        )
        assert updated.status_code == 200
        assert updated.json()["saved"] is True

        after_update = client.post("/v1/businesses/demo-store/forecast")
        assert after_update.status_code == 200
        assert pipeline.calls == 2


def test_api_rejects_short_history_bad_business_id_and_outside_date(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    app = create_app(
        db_path=tmp_path / "revenue.sqlite3",
        model_path=model_path,
        pipeline_override=ApiMockPipeline(),
    )

    with TestClient(app) as client:
        imported = client.post(
            "/v1/businesses/short/revenue/import",
            files={"file": ("revenue.csv", _csv(27), "text/csv")},
        )
        assert imported.status_code == 200
        assert client.post("/v1/businesses/short/forecast").status_code == 422
        assert client.get("/v1/businesses/bad.id/revenue").status_code == 422

        client.post(
            "/v1/businesses/full/revenue/import",
            files={"file": ("revenue.csv", _csv(28), "text/csv")},
        )
        outside = client.get("/v1/businesses/full/forecast/2026-02-20")
        assert outside.status_code == 404
