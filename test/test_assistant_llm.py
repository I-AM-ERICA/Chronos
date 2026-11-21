from __future__ import annotations

from datetime import date

import requests

from chronos_sales.assistant_forecast import ForecastPoint
from chronos_sales.assistant_llm import HyperClovaClient, safe_forecast_message


POINT = ForecastPoint(
    base_date=date(2026, 11, 19),
    forecast_date=date(2026, 12, 4),
    days_ahead=15,
    pred_q10=1_190_000,
    pred_q50=1_530_000,
    pred_q90=1_880_000,
)


class FakeResponse:
    def __init__(self, message: str, *, fail: bool = False):
        self.message = message
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise requests.HTTPError("401 Unauthorized")

    def json(self):
        return {
            "status": {"code": "20000"},
            "result": {"message": {"content": self.message}},
        }


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.request = None

    def post(self, url, **kwargs):
        self.request = {"url": url, **kwargs}
        if self.error:
            raise self.error
        return self.response


def test_hyperclova_valid_response_and_payload_contains_no_history():
    expected = safe_forecast_message(POINT)
    session = FakeSession(FakeResponse(expected))
    result = HyperClovaClient("secret", session=session).generate(POINT)
    assert result.message == expected
    assert result.source == "hyperclova"
    payload_text = str(session.request["json"])
    assert "1,190,000" not in payload_text
    assert "1,880,000" not in payload_text
    assert "history" not in payload_text.lower()
    assert "1,530,000" in payload_text


def test_changed_number_is_replaced_with_safe_template():
    session = FakeSession(
        FakeResponse("15일 뒤 12월 4일 예상 매출액은 9,999원입니다.")
    )
    result = HyperClovaClient("secret", session=session).generate(POINT)
    assert result.source == "template"
    assert result.message == safe_forecast_message(POINT)


def test_extra_management_advice_is_replaced_with_safe_template():
    message = safe_forecast_message(POINT) + " 재고를 늘리세요."
    result = HyperClovaClient(
        "secret", session=FakeSession(FakeResponse(message))
    ).generate(POINT)
    assert result.source == "template"
    assert result.message == safe_forecast_message(POINT)


def test_missing_key_auth_failure_and_timeout_use_safe_template():
    clients = [
        HyperClovaClient(None),
        HyperClovaClient("secret", session=FakeSession(FakeResponse("", fail=True))),
        HyperClovaClient(
            "secret", session=FakeSession(error=requests.Timeout("timed out"))
        ),
    ]
    for client in clients:
        result = client.generate(POINT)
        assert result.source == "template"
        assert result.message == safe_forecast_message(POINT)
