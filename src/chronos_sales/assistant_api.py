from __future__ import annotations

import os
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Path as ApiPath, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .assistant_forecast import FORECAST_DAYS, ForecastPoint, load_chronos_pipeline
from .assistant_service import ForecastRun, RevenueAssistantService
from .assistant_storage import MIN_HISTORY_DAYS, RevenueDataError
from .config import PROJECT_ROOT


BUSINESS_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "release"
    / "chronos-online-retail-v0.1"
    / "inference"
)
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "assistant" / "revenue.sqlite3"


class RevenueInput(BaseModel):
    revenue: int = Field(ge=0, le=9_223_372_036_854_775_807)


class PipelineHolder:
    """Load the large Chronos checkpoint once, on the first forecast request."""

    def __init__(self, model_path: Path, override: Any = None) -> None:
        self.model_path = model_path
        self.pipeline = override
        self._lock = threading.Lock()

    def get(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline
        with self._lock:
            if self.pipeline is None:
                if not self.model_path.is_dir():
                    raise FileNotFoundError(
                        f"Chronos 모델 폴더를 찾을 수 없습니다: {self.model_path}"
                    )
                self.pipeline = load_chronos_pipeline(self.model_path)
        return self.pipeline


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "ASSISTANT_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8501",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _point_payload(point: ForecastPoint) -> dict[str, str | int]:
    return point.as_api_dict()


def create_app(
    *,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    pipeline_override: Any = None,
) -> FastAPI:
    db_value: Path | str = (
        db_path
        if db_path is not None
        else os.getenv("REVENUE_DB_PATH", str(DEFAULT_DB_PATH))
    )
    model_value: Path | str = (
        model_path
        if model_path is not None
        else os.getenv("CHRONOS_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    )
    resolved_db = Path(db_value).expanduser()
    resolved_model = Path(model_value).expanduser()

    api = FastAPI(
        title="Chronos 매출 AI 비서 API",
        version="0.1.0",
        description=(
            f"사업장의 최근 {MIN_HISTORY_DAYS}일 일매출로 "
            f"다음 {FORECAST_DAYS}일 매출을 예측합니다."
        ),
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["*"],
    )
    api.state.db_path = resolved_db
    api.state.model_path = resolved_model
    api.state.pipeline_holder = PipelineHolder(resolved_model, pipeline_override)
    api.state.forecast_cache = {}
    api.state.cache_lock = threading.Lock()

    def service(business_id: str) -> RevenueAssistantService:
        if re.fullmatch(BUSINESS_ID_PATTERN, business_id) is None:
            raise HTTPException(
                status_code=422,
                detail="business_id는 1~64자의 영문·숫자·_·-만 사용할 수 있습니다.",
            )
        return RevenueAssistantService(resolved_db, resolved_model, business_id)

    def cached_forecast(current: RevenueAssistantService) -> ForecastRun:
        try:
            cache_key = current.current_cache_key()
            with api.state.cache_lock:
                cached = api.state.forecast_cache.get(cache_key)
            if cached is not None:
                return cached
            pipeline = api.state.pipeline_holder.get()
            run = current.run_forecast(pipeline)
            with api.state.cache_lock:
                api.state.forecast_cache[run.cache_key] = run
            return run
        except RevenueDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"예측 실행 실패: {exc}") from exc

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path_exists": resolved_model.is_dir(),
            "model_loaded": api.state.pipeline_holder.pipeline is not None,
            "minimum_history_days": MIN_HISTORY_DAYS,
            "forecast_days": FORECAST_DAYS,
        }

    @api.get("/v1/businesses/{business_id}/revenue")
    def list_revenue(
        business_id: str = ApiPath(pattern=BUSINESS_ID_PATTERN),
    ) -> dict[str, Any]:
        records = service(business_id).records()
        return {
            "business_id": business_id,
            "count": len(records),
            "records": [
                {
                    "date": record.date.isoformat(),
                    "revenue": record.revenue,
                    "source": record.source,
                }
                for record in records
            ],
        }

    @api.post("/v1/businesses/{business_id}/revenue/import")
    def import_revenue_csv(
        business_id: str = ApiPath(pattern=BUSINESS_ID_PATTERN),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=422, detail="CSV 파일만 업로드할 수 있습니다.")
        try:
            result = service(business_id).import_csv(file.file.read())
        except RevenueDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with api.state.cache_lock:
            api.state.forecast_cache.clear()
        return {
            "business_id": business_id,
            "inserted": result.inserted,
            "conflicts": [day.isoformat() for day in result.conflicts],
        }

    @api.put("/v1/businesses/{business_id}/revenue/{revenue_date}")
    def save_revenue(
        payload: RevenueInput,
        business_id: str = ApiPath(pattern=BUSINESS_ID_PATTERN),
        revenue_date: date = ApiPath(),
    ) -> dict[str, Any]:
        try:
            service(business_id).save_manual(revenue_date, payload.revenue)
        except RevenueDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with api.state.cache_lock:
            api.state.forecast_cache.clear()
        return {
            "business_id": business_id,
            "date": revenue_date.isoformat(),
            "revenue": payload.revenue,
            "saved": True,
        }

    @api.post("/v1/businesses/{business_id}/forecast")
    def create_forecast(
        business_id: str = ApiPath(pattern=BUSINESS_ID_PATTERN),
    ) -> dict[str, Any]:
        run = cached_forecast(service(business_id))
        return {
            "business_id": business_id,
            "context_days": MIN_HISTORY_DAYS,
            "forecast_days": FORECAST_DAYS,
            "points": [_point_payload(point) for point in run.points],
        }

    @api.get("/v1/businesses/{business_id}/forecast/{forecast_date}")
    def forecast_for_date(
        business_id: str = ApiPath(pattern=BUSINESS_ID_PATTERN),
        forecast_date: date = ApiPath(),
    ) -> dict[str, Any]:
        current = service(business_id)
        run = cached_forecast(current)
        try:
            point = current.select(run.points, forecast_date)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        answer = current.answer(point, os.getenv("CLOVA_STUDIO_API_KEY"))
        return {
            "business_id": business_id,
            "forecast": _point_payload(point),
            "message": answer.message,
            "message_source": answer.source,
        }

    return api


app = create_app()
