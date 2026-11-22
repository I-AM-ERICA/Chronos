from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from streamlit.testing.v1 import AppTest

from chronos_sales.assistant_app import _build_forecast_chart
from chronos_sales.assistant_forecast import forecast_with_pipeline
from chronos_sales.assistant_storage import RevenueStore


class AppMockPipeline:
    def predict(self, contexts, prediction_length, num_samples):
        values = np.tile(
            np.arange(1_000_000, 1_000_000 + prediction_length),
            (num_samples, 1),
        )
        return torch.tensor(values[None, :, :], dtype=torch.float32)


def test_streamlit_registered_revenue_to_selected_forecast_sentence(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "assistant.sqlite3"
    model_path = tmp_path / "inference"
    model_path.mkdir()
    start = date(2026, 1, 1)
    csv_text = "date,revenue\n" + "\n".join(
        f"{(start + timedelta(days=index)).isoformat()},{100_000 + index}"
        for index in range(60)
    )
    # Streamlit's test API does not currently set file_uploader values. The same
    # CSV registration service used by the UI is therefore invoked before render.
    RevenueStore(db_path).import_csv(csv_text)
    monkeypatch.setenv("REVENUE_DB_PATH", str(db_path))
    monkeypatch.setenv("CHRONOS_MODEL_PATH", str(model_path))
    monkeypatch.delenv("CLOVA_STUDIO_API_KEY", raising=False)

    # Resolve the entry point from the checked-out test file. PROJECT_ROOT is a
    # runtime data-path default and can point at the Python installation when
    # the package is installed through an editable wheel in CI.
    app_path = Path(__file__).resolve().parents[1] / "assistant_app.py"
    with patch(
        "chronos_sales.assistant_app.cached_pipeline", return_value=AppMockPipeline()
    ):
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        assert not app.exception
        forecast_button = next(
            button
            for button in app.button
            if button.label == "다음 14일 매출 알아보기"
        )
        forecast_button.click()
        app.run()
        assert not app.exception

    forecast_date = next(
        item
        for item in app.date_input
        if item.label == "언제의 예상 매출을 보시겠어요?"
    )
    forecast_date.set_value(date(2026, 3, 11))
    app.run()
    assert not app.exception
    rendered = "\n".join(markdown.value for markdown in app.markdown)
    assert "10일 뒤 3월 11일 예상 매출액은 1,000,009원입니다." in rendered

    RevenueStore(db_path).upsert_manual(date(2026, 2, 1), 999_999)
    app.run()
    assert any("매출 데이터가 바뀌었습니다" in item.value for item in app.info)


def test_forecast_chart_contains_daily_bars_labels_and_selected_highlight(tmp_path):
    store = RevenueStore(tmp_path / "chart.sqlite3")
    start = date(2026, 1, 1)
    store.import_csv(
        "date,revenue\n"
        + "\n".join(
            f"{(start + timedelta(days=index)).isoformat()},{100_000 + index}"
            for index in range(60)
        )
    )
    records = store.list_records()
    forecasts = forecast_with_pipeline(AppMockPipeline(), records)
    spec = _build_forecast_chart(records, forecasts).to_dict()
    assert len(spec["layer"]) == 2
    assert spec["layer"][0]["mark"]["type"] == "bar"
    assert spec["layer"][1]["mark"]["type"] == "text"
    assert spec["layer"][0]["encoding"]["color"]["condition"]["test"] == (
        "datum.is_selected"
    )
