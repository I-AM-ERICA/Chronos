from __future__ import annotations

from datetime import date

import requests


API_URL = "http://localhost:8000"
BUSINESS_ID = "demo-store"


def main() -> None:
    with open("sample_revenue_60days.csv", "rb") as csv_file:
        imported = requests.post(
            f"{API_URL}/v1/businesses/{BUSINESS_ID}/revenue/import",
            files={"file": ("revenue.csv", csv_file, "text/csv")},
            timeout=30,
        )
    imported.raise_for_status()
    print(imported.json())

    forecast = requests.post(
        f"{API_URL}/v1/businesses/{BUSINESS_ID}/forecast", timeout=120
    )
    forecast.raise_for_status()
    points = forecast.json()["points"]
    print(f"{len(points)}일 예측 완료")

    selected_date = date.fromisoformat(points[0]["forecast_date"])
    answer = requests.get(
        f"{API_URL}/v1/businesses/{BUSINESS_ID}/forecast/{selected_date.isoformat()}",
        timeout=30,
    )
    answer.raise_for_status()
    print(answer.json()["message"])


if __name__ == "__main__":
    main()
