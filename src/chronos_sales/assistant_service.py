from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .assistant_forecast import (
    ForecastPoint,
    forecast_cache_key,
    forecast_with_pipeline,
    select_forecast,
)
from .assistant_llm import HyperClovaClient, MessageResult
from .assistant_storage import (
    ImportResult,
    RevenueRecord,
    RevenueStore,
    validate_forecast_history,
)


@dataclass(frozen=True)
class ForecastRun:
    cache_key: str
    points: list[ForecastPoint]


class RevenueAssistantService:
    """Framework-independent facade for Streamlit, FastAPI, or another UI."""

    def __init__(
        self,
        db_path: Path | str,
        model_path: Path | str,
        business_id: str = "default",
    ) -> None:
        self.store = RevenueStore(db_path)
        self.model_path = Path(model_path).expanduser()
        self.business_id = business_id

    def import_csv(self, raw: bytes | str) -> ImportResult:
        return self.store.import_csv(raw, self.business_id)

    def save_manual(self, day: date, revenue: int) -> None:
        self.store.upsert_manual(day, revenue, self.business_id)

    def records(self) -> list[RevenueRecord]:
        return self.store.list_records(self.business_id)

    def forecast_context(self) -> list[RevenueRecord]:
        return validate_forecast_history(self.records())

    def current_cache_key(self) -> str:
        return forecast_cache_key(self.records(), self.model_path)

    def run_forecast(self, pipeline: Any) -> ForecastRun:
        records = self.records()
        return ForecastRun(
            cache_key=forecast_cache_key(records, self.model_path),
            points=forecast_with_pipeline(pipeline, records),
        )

    @staticmethod
    def select(points: list[ForecastPoint], selected_date: date) -> ForecastPoint:
        return select_forecast(points, selected_date)

    @staticmethod
    def answer(
        point: ForecastPoint, api_key: str | None, *, session: Any = None
    ) -> MessageResult:
        client = (
            HyperClovaClient(api_key)
            if session is None
            else HyperClovaClient(api_key, session=session)
        )
        return client.generate(point)
