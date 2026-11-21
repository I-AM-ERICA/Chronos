from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from .assistant_forecast import (
    FORECAST_DAYS,
    ForecastPoint,
    load_chronos_pipeline,
)
from .assistant_service import RevenueAssistantService
from .assistant_storage import MIN_HISTORY_DAYS, RevenueDataError
from .config import PROJECT_ROOT


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "release"
    / "chronos-online-retail-v0.1"
    / "inference"
)
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "assistant" / "revenue.sqlite3"

APP_CSS = """
<style>
  .stApp { background: #f4f7fb; }
  .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 4rem; }
  .hero-card {
    padding: 2rem 2.2rem; border-radius: 24px;
    background: linear-gradient(135deg, #2f6df6 0%, #5687ff 100%);
    box-shadow: 0 14px 32px rgba(47, 109, 246, 0.18); color: white;
    margin-bottom: 1.4rem;
  }
  .hero-card .eyebrow { font-size: 0.88rem; font-weight: 700; opacity: 0.9; }
  .hero-card h1 { margin: 0.35rem 0 0.55rem; font-size: 2rem; line-height: 1.25; }
  .hero-card p { margin: 0; font-size: 1.02rem; opacity: 0.94; }
  .result-card {
    padding: 1.6rem 1.7rem; border-radius: 22px;
    background: linear-gradient(145deg, #ffffff 0%, #f3f7ff 100%);
    border: 1px solid #dbe7ff; box-shadow: 0 12px 30px rgba(37, 91, 215, 0.10);
    margin: 0.8rem 0 1rem;
  }
  .result-card .badge {
    display: inline-block; padding: 0.35rem 0.7rem; border-radius: 999px;
    color: #245bd7; background: #e8f0ff; font-size: 0.78rem; font-weight: 800;
  }
  .result-card .label { color: #475569; font-size: 0.95rem; font-weight: 700; margin-top: 0.85rem; }
  .result-card .amount { color: #245bd7; font-size: 2.5rem; font-weight: 850; margin: 0.25rem 0 0.45rem; letter-spacing: -0.04em; }
  .result-card .sentence { color: #334155; font-size: 1.04rem; margin: 0; line-height: 1.65; }
  .range-title { color: #334155; font-size: 0.95rem; font-weight: 800; margin: 1.2rem 0 0.45rem; }
  div[data-testid="stMetric"] {
    background: white; border: 1px solid #e5eaf2; border-radius: 16px;
    padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
  }
  div[data-testid="stMetric"] label { color: #64748b; }
  div[data-testid="stMetricValue"] { color: #1e3a5f; font-weight: 800; }
  #vg-tooltip-element.vg-tooltip,
  .vg-tooltip {
    font-size: 15px !important;
    line-height: 1.55 !important;
    padding: 12px 14px !important;
    border-radius: 12px !important;
    border: 1px solid #d7e1f0 !important;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18) !important;
    color: #1e293b !important;
    background: rgba(255, 255, 255, 0.98) !important;
  }
  #vg-tooltip-element.vg-tooltip td,
  #vg-tooltip-element.vg-tooltip th,
  .vg-tooltip td,
  .vg-tooltip th {
    font-size: 15px !important;
    padding: 4px 7px !important;
  }
  #vg-tooltip-element.vg-tooltip td.value,
  .vg-tooltip td.value {
    color: #1d4ed8 !important;
    font-weight: 800 !important;
  }
  div[data-testid="stFileUploaderDropzone"] { border-radius: 16px; background: white; }
  .stButton > button { border-radius: 12px; min-height: 46px; font-weight: 700; }
  .stButton > button[kind="primary"] { background: #356cf6; border-color: #356cf6; }
  div[data-baseweb="tab-list"] { gap: 0.5rem; }
  button[data-baseweb="tab"] { border-radius: 12px 12px 0 0; font-weight: 700; }
  @media (max-width: 640px) {
    .block-container { padding: 1rem 0.8rem 3rem; }
    .hero-card { padding: 1.5rem 1.25rem; border-radius: 19px; }
    .hero-card h1 { font-size: 1.55rem; }
    .result-card .amount { font-size: 1.65rem; }
  }
</style>
"""


def configured_model_path() -> Path:
    return Path(os.getenv("CHRONOS_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()


def configured_db_path() -> Path:
    return Path(os.getenv("REVENUE_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()


@st.cache_resource(show_spinner="Chronos 모델을 불러오는 중입니다...")
def cached_pipeline(model_path: str) -> Any:
    return load_chronos_pipeline(model_path)


def _records_frame(service: RevenueAssistantService) -> pd.DataFrame:
    records = service.records()
    return pd.DataFrame(
        [
            {
                "날짜": record.date,
                "매출액(KRW)": record.revenue,
                "등록 방식": record.source,
            }
            for record in records
        ]
    )


def _render_registration(service: RevenueAssistantService) -> None:
    st.subheader("매출 내역을 알려주세요")
    st.write("엑셀에서 `date,revenue` 형식의 CSV로 저장한 파일을 올려주세요.")
    uploaded = st.file_uploader(
        "매출 CSV 파일 선택", type=["csv"], key="revenue_csv"
    )
    if st.button("파일 내용 등록하기", disabled=uploaded is None, use_container_width=True):
        assert uploaded is not None
        try:
            result = service.import_csv(uploaded.getvalue())
            st.success(f"매출 내역 {result.inserted}일을 등록했습니다.")
            if result.conflicts:
                conflict_text = ", ".join(day.isoformat() for day in result.conflicts)
                st.warning(
                    "기존 날짜는 덮어쓰지 않았습니다. "
                    f"수동 입력으로 수정하세요: {conflict_text}"
                )
        except RevenueDataError as exc:
            st.error(str(exc))

    with st.expander("하루 매출을 직접 입력하거나 고치기"):
        left, right = st.columns(2)
        manual_date = left.date_input("날짜", value=date.today(), key="manual_date")
        manual_revenue = right.number_input(
            "매출액(원)", min_value=0, step=1_000, value=0, key="manual_revenue"
        )
        if st.button("이 날짜 매출 저장", use_container_width=True):
            service.save_manual(manual_date, int(manual_revenue))
            st.success(f"{manual_date.isoformat()} 매출을 저장했습니다.")

    frame = _records_frame(service)
    if not frame.empty:
        first_day = frame["날짜"].min().isoformat()
        last_day = frame["날짜"].max().isoformat()
        st.success(f"현재 {len(frame)}일 등록 · {first_day} ~ {last_day}")
        with st.expander("등록된 매출 내역 확인"):
            st.dataframe(
                frame.tail(60).sort_values("날짜", ascending=False),
                hide_index=True,
                use_container_width=True,
            )


def _get_or_run_forecast(
    service: RevenueAssistantService, pipeline: Any
) -> list[ForecastPoint]:
    cache_key = service.current_cache_key()
    cache = st.session_state.setdefault("forecast_cache", {})
    if cache_key not in cache:
        cache.clear()
        cache[cache_key] = service.run_forecast(pipeline).points
    return cache[cache_key]


def _compact_won(amount: int) -> str:
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:.1f}억"
    if amount >= 10_000:
        return f"{amount / 10_000:.0f}만"
    return f"{amount:,}"


def _build_forecast_chart(
    records: list[Any],
    forecasts: list[ForecastPoint],
    selected_date: date | None = None,
) -> alt.LayerChart:
    del records  # The chart intentionally focuses on the actionable 14-day forecast.
    selected_date = selected_date or forecasts[0].forecast_date
    forecast_frame = pd.DataFrame(
        {
            "date": [point.forecast_date for point in forecasts],
            "date_label": [
                f"{point.forecast_date.month}/{point.forecast_date.day}"
                for point in forecasts
            ],
            "order": list(range(len(forecasts))),
            "is_selected": [
                point.forecast_date == selected_date for point in forecasts
            ],
            "pred_q10": [point.pred_q10 for point in forecasts],
            "pred_q50": [point.pred_q50 for point in forecasts],
            "pred_q90": [point.pred_q90 for point in forecasts],
            "pred_q50_manwon": [point.pred_q50 / 10_000 for point in forecasts],
            "amount_label": [_compact_won(point.pred_q50) for point in forecasts],
        }
    )
    x_axis = alt.X(
        "date_label:N",
        title=None,
        sort=alt.SortField("order"),
        axis=alt.Axis(labelAngle=0, labelColor="#64748b", labelPadding=10),
    )
    bars = (
        alt.Chart(forecast_frame)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=32)
        .encode(
            x=x_axis,
            y=alt.Y(
                "pred_q50_manwon:Q",
                title="예상 매출(만원)",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(gridColor="#edf1f7", tickCount=5),
            ),
            color=alt.condition(
                "datum.is_selected",
                alt.value("#356cf6"),
                alt.value("#cbd9f4"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="예측 날짜", format="%Y-%m-%d"),
                alt.Tooltip("pred_q50:Q", title="예상 매출", format=","),
                alt.Tooltip("pred_q10:Q", title="낮게 예상하면", format=","),
                alt.Tooltip("pred_q90:Q", title="높게 예상하면", format=","),
            ],
        )
    )
    labels = (
        alt.Chart(forecast_frame)
        .mark_text(dy=-9, color="#475569", fontSize=11, fontWeight=700)
        .encode(
            x=x_axis,
            y=alt.Y("pred_q50_manwon:Q"),
            text="amount_label:N",
        )
    )
    return (
        (bars + labels)
        .properties(
            title=f"앞으로 {FORECAST_DAYS}일 예상 매출",
            height=360,
        )
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#e9eef6", labelColor="#64748b", titleColor="#475569")
        .configure_title(anchor="start", color="#1e293b", fontSize=17, fontWeight=700)
    )


def _render_forecast_chart(
    records: list[Any], forecasts: list[ForecastPoint], selected_date: date
) -> None:
    chart = _build_forecast_chart(records, forecasts, selected_date)
    st.altair_chart(chart, use_container_width=True)
    st.caption("파란 막대는 달력에서 선택한 날짜입니다. 막대에 마우스를 올리면 예상 범위도 볼 수 있어요.")


def _render_forecast(service: RevenueAssistantService) -> None:
    st.subheader(f"앞으로 {FORECAST_DAYS}일 매출을 확인하세요")
    st.info(
        "현재는 시험용 예측 모델입니다. 국내 매장 데이터로 "
        "추가 검증하기 전까지 중요한 경영 결정에는 참고로만 사용하세요."
    )
    records = service.records()
    try:
        context = service.forecast_context()
        st.success(
            f"준비 완료 · 전체 {len(records)}일 중 최근 "
            f"{len(context)}일을 예측에 사용합니다."
        )
    except RevenueDataError as exc:
        st.info(str(exc))
        return
    if not service.model_path.is_dir():
        st.error(
            f"모델 폴더를 찾을 수 없습니다: {service.model_path}\n\n"
            "CHRONOS_MODEL_PATH를 추론용 모델 압축 해제 폴더로 설정하세요."
        )
        return

    current_cache_key = service.current_cache_key()

    if st.button(
        f"다음 {FORECAST_DAYS}일 매출 알아보기",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("매출 흐름을 살펴보는 중입니다..."):
                pipeline = cached_pipeline(str(service.model_path.resolve()))
                st.session_state["active_forecasts"] = _get_or_run_forecast(
                    service, pipeline
                )
                st.session_state["active_forecast_key"] = current_cache_key
            st.success(f"다음 {FORECAST_DAYS}일 예측이 준비됐습니다.")
        except Exception as exc:
            st.error(f"예측 실패: {exc}")
            return

    forecasts: list[ForecastPoint] | None = st.session_state.get("active_forecasts")
    if not forecasts:
        return
    forecast_is_stale = (
        forecasts[0].base_date != context[-1].date
        or st.session_state.get("active_forecast_key") != current_cache_key
    )
    if forecast_is_stale:
        st.info(
            f"매출 데이터가 바뀌었습니다. "
            f"{FORECAST_DAYS}일 매출 예측을 다시 실행하세요."
        )
        return
    selected_date = st.date_input(
        "언제의 예상 매출을 보시겠어요?",
        value=forecasts[0].forecast_date,
        min_value=forecasts[0].forecast_date,
        max_value=forecasts[-1].forecast_date,
        key="forecast_date",
    )
    point = service.select(forecasts, selected_date)
    message_cache = st.session_state.setdefault("message_cache", {})
    message_key = (
        point.forecast_date.isoformat(),
        point.days_ahead,
        point.pred_q50,
        bool(os.getenv("CLOVA_STUDIO_API_KEY")),
    )
    if message_key not in message_cache:
        message_cache[message_key] = service.answer(
            point, os.getenv("CLOVA_STUDIO_API_KEY")
        )
    result = message_cache[message_key]
    st.markdown(
        f"""
        <div class="result-card">
          <span class="badge">AI 매출 예측</span>
          <div class="label">{point.forecast_date.month}월 {point.forecast_date.day}일 예상 매출</div>
          <div class="amount">{point.pred_q50:,}원</div>
          <p class="sentence">{result.message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="range-title">이 날의 예상 범위</div>', unsafe_allow_html=True)
    low, likely, high = st.columns(3)
    low.metric("낮게 예상하면", f"{point.pred_q10:,}원")
    likely.metric("가장 가능성 높은 금액", f"{point.pred_q50:,}원")
    high.metric("높게 예상하면", f"{point.pred_q90:,}원")

    st.markdown("#### 매출 흐름")
    _render_forecast_chart(records, forecasts, selected_date)
    with st.expander("개발자용 예측 정보"):
        st.json(point.as_api_dict())
        st.caption(
            f"문장 생성 방식: "
            f"{'HyperCLOVA X' if result.source == 'hyperclova' else '안전 템플릿'}"
        )


def main() -> None:
    st.set_page_config(
        page_title="우리 가게 매출 비서",
        page_icon="📈",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="eyebrow">AI 우리 가게 매출 비서</div>
          <h1>우리 가게, 앞으로 얼마나 벌까요?</h1>
          <p>최근 {MIN_HISTORY_DAYS}일 매출을 바탕으로 다음 {FORECAST_DAYS}일 예상 매출을 날짜별로 확인해 보세요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    service = RevenueAssistantService(configured_db_path(), configured_model_path())
    registration_tab, forecast_tab = st.tabs(["1. 매출 입력", "2. 매출 예측"])
    with registration_tab:
        _render_registration(service)
    with forecast_tab:
        _render_forecast(service)


if __name__ == "__main__":
    main()
