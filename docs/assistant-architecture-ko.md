# 매출 AI 비서 화면 교체 가이드

## 분리 원칙

배포 화면과 예측 기능을 다음과 같이 분리합니다.

```text
사용자 화면(Streamlit, React, 모바일 앱)
                    ↓ HTTP/JSON
              FastAPI 어댑터
                    ↓
        RevenueAssistantService
        ↓           ↓           ↓
SQLite 매출     Chronos 예측    HyperCLOVA 문장
```

- `assistant_app.py`: Streamlit 화면만 담당합니다.
- `assistant_service.py`: 화면과 무관하게 매출 등록, 예측, 날짜 선택, 문장 생성을 연결합니다.
- `assistant_storage.py`: CSV 검증과 SQLite 저장을 담당합니다.
- `assistant_forecast.py`: Chronos 모델 로드와 최근 28일 입력 기반 14일 예측을 담당합니다.
- `assistant_llm.py`: HyperCLOVA X 호출과 안전 템플릿을 담당합니다.
- `assistant_api.py`: React·Vue·모바일 앱이 호출할 REST API를 제공합니다.

## FastAPI 실행

```bash
pip install -e ".[assistant,api]"
uvicorn chronos_sales.assistant_api:app --host 0.0.0.0 --port 8000
```

실행 후 자동 API 문서는 다음 주소에서 확인합니다.

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- 상태 확인: `http://localhost:8000/health`

Chronos 모델은 첫 예측 요청 때 한 번만 로드되고 서버 메모리에서 재사용됩니다.
동일한 데이터와 모델의 예측 결과도 캐시됩니다.

## REST API

| 메서드 | 경로 | 기능 |
|---|---|---|
| `GET` | `/health` | 서버·모델 상태 확인 |
| `GET` | `/v1/businesses/{business_id}/revenue` | 등록된 일매출 조회 |
| `POST` | `/v1/businesses/{business_id}/revenue/import` | `date,revenue` CSV 등록 |
| `PUT` | `/v1/businesses/{business_id}/revenue/{date}` | 하루 매출 등록·수정 |
| `POST` | `/v1/businesses/{business_id}/forecast` | 다음 14일 예측 |
| `GET` | `/v1/businesses/{business_id}/forecast/{date}` | 선택 날짜 예측과 문장 조회 |

`business_id`는 영문, 숫자, `_`, `-`로 구성한 1~64자 식별자입니다.

### CSV 등록

```bash
curl -X POST \
  -F "file=@sample_revenue_60days.csv;type=text/csv" \
  http://localhost:8000/v1/businesses/demo-store/revenue/import
```

### 14일 예측

```bash
curl -X POST \
  http://localhost:8000/v1/businesses/demo-store/forecast
```

### 선택 날짜 조회

```bash
curl \
  http://localhost:8000/v1/businesses/demo-store/forecast/2026-11-22
```

응답 예시:

```json
{
  "business_id": "demo-store",
  "forecast": {
    "base_date": "2026-11-18",
    "forecast_date": "2026-11-22",
    "days_ahead": 4,
    "pred_q10": 379777,
    "pred_q50": 533777,
    "pred_q90": 604644,
    "currency": "KRW"
  },
  "message": "4일 뒤 11월 22일 예상 매출액은 533,777원입니다.",
  "message_source": "template"
}
```

## React·JavaScript 호출 예시

```javascript
const response = await fetch(
  "http://localhost:8000/v1/businesses/demo-store/forecast",
  { method: "POST" },
);
const result = await response.json();
console.log(result.points);
```

API 호출 예제 파일은 `examples/assistant_api_client.py`에도 제공합니다.

## Docker 실행

`.env.example`을 `.env`로 복사한 뒤 모델·DB 폴더를 볼륨으로 연결합니다.

```bash
docker build -t chronos-sales-assistant .
docker run --rm -p 8000:8000 --env-file .env \
  -v /absolute/model/inference:/models/chronos:ro \
  -v /absolute/assistant-data:/app/data/assistant \
  -e CHRONOS_MODEL_PATH=/models/chronos \
  chronos-sales-assistant
```

`ASSISTANT_CORS_ORIGINS`에는 파일을 호출할 프론트엔드 주소만 쉼표로 나누어 등록합니다.

## 다른 화면으로 옮기는 방법

Streamlit 코드를 복사하지 말고 `RevenueAssistantService`만 호출합니다.
예를 들어 FastAPI 또는 기존 AI 비서 백엔드에서는 다음 순서를 사용합니다.

```python
service = RevenueAssistantService(db_path, model_path)
service.import_csv(csv_bytes)
run = service.run_forecast(loaded_chronos_pipeline)
point = service.select(run.points, selected_date)
answer = service.answer(point, clova_api_key)

response = {
    "forecast": point.as_api_dict(),
    "message": answer.message,
    "message_source": answer.source,
}
```

프론트엔드는 이 JSON만 받아 금액 카드, 달력, 그래프로 표시합니다.
Chronos 모델은 백엔드 시작 시 한 번만 로드하고 재사용합니다.

## 교체하지 않는 규칙

- 날짜, 일수, 금액은 코드가 확정합니다.
- LLM은 표현만 담당하며 숫자를 바꾸면 안전 템플릿으로 대체합니다.
- 데이터 해시와 모델 식별자가 달라지면 이전 예측을 재사용하지 않습니다.
- API 키, 사업장 데이터, 예측 캐시는 화면에 하드코딩하지 않습니다.
