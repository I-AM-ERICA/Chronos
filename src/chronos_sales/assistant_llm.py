from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from .assistant_forecast import ForecastPoint


CLOVA_API_URL = (
    "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002"
)


@dataclass(frozen=True)
class MessageResult:
    message: str
    source: str
    error: str | None = None


def safe_forecast_message(point: ForecastPoint) -> str:
    return (
        f"{point.days_ahead}일 뒤 {point.forecast_date.month}월 "
        f"{point.forecast_date.day}일 예상 매출액은 "
        f"{point.pred_q50:,}원입니다."
    )


def _valid_message(message: str, point: ForecastPoint) -> bool:
    stripped = message.strip()
    if not stripped or "\n" in stripped or "\r" in stripped or len(stripped) > 120:
        return False
    message = " ".join(stripped.split())
    # Accept only the promised one-sentence format. This also prevents the LLM
    # from appending unsupported management advice without introducing numbers.
    return message == safe_forecast_message(point)


class HyperClovaClient:
    def __init__(
        self,
        api_key: str | None,
        *,
        session: Any = requests,
        timeout: tuple[float, float] = (5.0, 15.0),
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.session = session
        self.timeout = timeout

    @staticmethod
    def build_payload(point: ForecastPoint) -> dict[str, Any]:
        facts = (
            f"예측일은 기준일로부터 {point.days_ahead}일 뒤이고, "
            f"표시 날짜는 {point.forecast_date.month}월 "
            f"{point.forecast_date.day}일이며, 예상 매출액은 "
            f"{point.pred_q50:,}원이다."
        )
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "제공된 일수·날짜·금액을 절대 바꾸지 말고, "
                        "한국어 한 문장으로만 답하라. 조언과 다른 숫자를 추가하지 말라."
                    ),
                },
                {"role": "user", "content": facts},
            ],
            "topP": 0.1,
            "topK": 1,
            "maxTokens": 64,
            "temperature": 0.0,
            "repetitionPenalty": 1.0,
            "seed": 42,
            "includeAiFilters": True,
        }

    def generate(self, point: ForecastPoint) -> MessageResult:
        fallback = safe_forecast_message(point)
        if not self.api_key:
            return MessageResult(fallback, "template", "CLOVA Studio API 키가 없습니다.")
        try:
            response = self.session.post(
                CLOVA_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=self.build_payload(point),
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            if str(body.get("status", {}).get("code")) != "20000":
                raise RuntimeError("응답 상태 코드가 20000이 아닙니다.")
            message = str(body["result"]["message"]["content"]).strip()
            if not _valid_message(message, point):
                raise ValueError("LLM이 확정된 날짜·일수·금액을 변경했습니다.")
            return MessageResult(message, "hyperclova")
        except (requests.RequestException, KeyError, TypeError, ValueError, RuntimeError) as exc:
            return MessageResult(fallback, "template", str(exc))
