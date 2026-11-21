from __future__ import annotations

from datetime import date, timedelta

import pytest

from chronos_sales.assistant_storage import (
    RevenueDataError,
    RevenueStore,
    parse_revenue_csv,
    validate_forecast_history,
)


def _csv(start: date, days: int) -> str:
    rows = ["date,revenue"]
    rows.extend(
        f"{(start + timedelta(days=index)).isoformat()},{1000 + index}"
        for index in range(days)
    )
    return "\n".join(rows)


def test_csv_import_reports_conflicts_and_manual_input_updates(tmp_path):
    store = RevenueStore(tmp_path / "revenue.sqlite3")
    first = store.import_csv("date,revenue\n2026-01-01,1000\n2026-01-02,2000")
    assert first.inserted == 2
    assert first.conflicts == ()

    second = store.import_csv("date,revenue\n2026-01-02,9999\n2026-01-03,3000")
    assert second.inserted == 1
    assert second.conflicts == (date(2026, 1, 2),)
    assert [record.revenue for record in store.list_records()] == [1000, 2000, 3000]

    store.upsert_manual(date(2026, 1, 2), 7777)
    records = store.list_records()
    assert [record.revenue for record in records] == [1000, 7777, 3000]
    assert records[1].source == "manual"


@pytest.mark.parametrize(
    "raw, message",
    [
        ("date,revenue\n2026-02-30,1", "존재하지"),
        ("date,revenue\n2026-01-01,-1", "0 이상"),
        ("date,revenue\n2026-01-01,1.5", "0 이상"),
        ("date,revenue\n2026-01-01,1\n2026-01-01,2", "중복"),
        ("when,amount\n2026-01-01,1", "date,revenue"),
        ("revenue,date\n1,2026-01-01", "date,revenue"),
    ],
)
def test_csv_validation_rejects_bad_rows(raw, message):
    with pytest.raises(RevenueDataError, match=message):
        parse_revenue_csv(raw)


def test_history_requires_28_consecutive_explicit_days_and_uses_latest_28(tmp_path):
    store = RevenueStore(tmp_path / "revenue.sqlite3")
    store.import_csv(_csv(date(2025, 1, 1), 27))
    with pytest.raises(RevenueDataError, match="27일"):
        validate_forecast_history(store.list_records())

    store.upsert_manual(date(2025, 1, 29), 1000)
    with pytest.raises(RevenueDataError, match="2025-01-28"):
        validate_forecast_history(store.list_records())

    long_store = RevenueStore(tmp_path / "long.sqlite3")
    long_store.import_csv(_csv(date(2024, 1, 1), 600))
    context = validate_forecast_history(long_store.list_records())
    assert len(context) == 28
    assert context[-1].date == date(2024, 1, 1) + timedelta(days=599)
