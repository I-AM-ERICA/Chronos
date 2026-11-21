# UCI Online Retail II 일매출 Chronos 예측

이 파이프라인은 UCI Online Retail II의 거래수량과 단가를 곱해 한 소매업체의 일별 순상품매출 추정치를 만들고 Chronos로 다음 28일을 예측합니다.

## 데이터와 라이선스

- 출처: UCI Machine Learning Repository, Online Retail II
- DOI: `10.24432/C5CG6D`
- 라이선스: CC BY 4.0
- 기간: 2009-12-01 ~ 2011-12-09
- 화폐: 영국 파운드(GBP)

데이터와 수정 결과를 공유하거나 경진대회에서 사용할 수 있지만 출처와 라이선스를 표시해야 합니다.

## 매출 정의

각 거래행에 대해 `Quantity × Price`를 계산합니다.

- 총상품매출: 양수 수량 거래만 합산
- 반품금액: 음수 수량 거래를 음수로 합산
- 순상품매출 추정치: 총상품매출 + 반품금액
- 취소 주문: Invoice가 `C`로 시작하는 행을 감사 정보에 기록
- 음수 가격: 정상 상품 판매가 아닌 조정행으로 보고 별도 CSV로 격리
- 정확 중복: 두 연도 시트의 겹치는 기간과 시트 내부 중복을 포함해 제거
- 거래가 없는 날짜: 연속 시계열 유지를 위해 매출 0으로 추가

세금, 배송료, 결제 수수료 등이 없으므로 회계상 순매출이 아니라 `net product revenue proxy`입니다.

## 데이터 준비

원본은 다음 위치에 저장되어 있습니다.

```text
data/raw/online_retail_ii/online_retail_II.xlsx
```

변환 명령:

```powershell
$env:PYTHONUTF8="1"
chronos-sales prepare-online-retail
```

주요 결과:

```text
data/online_retail_revenue/daily_revenue.csv
data/online_retail_revenue/daily_revenue.parquet
data/online_retail_revenue/transaction_audit.csv
data/online_retail_revenue/excluded_negative_price_rows.csv
data/online_retail_revenue/online_retail_manifest.csv
data/online_retail_revenue/arrow/train/data.arrow
```

## 예측과 학습

```powershell
# 공개 모델 CPU 제로샷
chronos-sales zero-shot --dataset online-retail-revenue --device cpu --batch-size 1

# CPU 10단계 실행 경로 확인
chronos-sales finetune --dataset online-retail-revenue --cpu --batch-size 1 --config-path configs/online-retail-smoke.yaml

# GPU 전체 1,000단계 미세조정
chronos-sales finetune --dataset online-retail-revenue --batch-size 32

# 검증 WQL 기준 체크포인트 선택 후 시험 평가
chronos-sales evaluate --dataset online-retail-revenue --device cuda --batch-size 1

# 그래프
chronos-sales plot --dataset online-retail-revenue --store-id ONLINE_RETAIL_UK

# 동일 분할에서 Random Forest, XGBoost, CatBoost 비교
pip install -e ".[tree-models]"
chronos-sales benchmark-models
```

학습 구간은 처음 683일, 검증은 다음 28일, 최종 시험은 마지막 28일입니다. 최종 시험 데이터는 Arrow 학습 파일과 체크포인트 선택에 포함되지 않습니다.

## 일반 머신러닝 모델 비교

`benchmark-models`는 Chronos와 동일한 학습·검증·시험 날짜를 사용합니다. 일반
회귀 모델에는 과거 1·2·3·7·14·21·28·56일 매출, 7·14·28·56일 이동통계,
요일·월 등의 달력 특성을 제공합니다. 미래 28일은 하루씩 재귀 예측하므로 시험
구간의 실제 매출이 입력 특성에 유출되지 않습니다.

주요 결과:

```text
outputs/online_retail_revenue/benchmarks/tabular_model_metrics.csv
outputs/online_retail_revenue/benchmarks/tabular_model_forecasts.csv
outputs/online_retail_revenue/benchmarks/all_model_comparison.csv
outputs/online_retail_revenue/benchmarks/benchmark_audit.json
```

## 3구간 롤링 모델 비교

동일한 28일 예측 조건에서 미세조정 Chronos, Random Forest, XGBoost,
CatBoost, AutoARIMA, AutoETS를 비교하고 모델 파일까지 저장합니다.

```bash
pip install -e ".[comparison-models]"
chronos-sales rolling-benchmark --stage prepare --folds 3
chronos-sales rolling-benchmark --stage chronos --device cuda --batch-size 4
chronos-sales rolling-benchmark --stage baselines
chronos-sales rolling-benchmark --stage summarize
```

결과는 `outputs/online_retail_revenue/rolling_benchmark/`에 저장됩니다. 세 Fold의
평균 WAPE를 1차 기준으로 하고, 동률이면 평균 MASE와 MAE 순으로 순위를 정합니다.

## 경진대회 설명 시 주의점

이 데이터는 영국의 한 온라인 소매업체 데이터이며 국내 오프라인 소상공인 POS 데이터가 아닙니다. 발표에서는 매출예측 기술 검증용 공개 데이터라고 표시하고, 실제 도입 시 같은 입력 형식으로 매장 POS 일매출을 연결한다고 설명합니다.
