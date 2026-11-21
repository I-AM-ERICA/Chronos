# UCI Online Retail II 일매출 예측 결과

이 폴더에는 동일한 3개 롤링 시험 구간에서 Chronos와 일반 예측 모델을 비교한 경량 결과물이 들어 있습니다.

## 실험 조건

- 데이터: UCI Online Retail II에서 집계한 일별 순상품매출 대용치(GBP)
- 예측 길이: Fold마다 28일
- 비교 모델: fine-tuned Chronos, Random Forest, XGBoost, CatBoost, AutoARIMA, AutoETS
- 기준선 모델은 학습 모델 최종 순위에서 제외했습니다.
- 1차 순위 기준: 3개 시험 Fold의 평균 WAPE

## 파일 설명

- `model_ranking.csv`: 평균 WAPE 기준 최종 모델 순위
- `summary_metrics.csv`: 모델별 평균·표준편차·최악 성능
- `fold_metrics.csv`: Fold와 모델별 WQL, MASE, MAE, WAPE
- `validation_candidates.csv`: 검증 구간에서 비교한 후보 설정과 점수
- `predictions.csv`: 날짜별 실제 매출과 분위수 예측값
- `comparison.png`: 모델 성능 비교 그래프
- `validity_decision.json`: Chronos 상위 3위 진입 여부 판정
- `run_manifest.json`: 데이터 해시, Fold 날짜, 학습 설정, 라이브러리 버전

## 주요 결과

| 순위 | 모델 | 평균 WAPE |
|---:|---|---:|
| 1 | XGBoost | 21.88% |
| 2 | Random Forest | 24.49% |
| 3 | CatBoost | 26.67% |
| 4 | Fine-tuned Chronos | 27.33% |
| 5 | AutoETS | 31.19% |
| 6 | AutoARIMA | 31.33% |

원본 거래 데이터, 가상환경, 학습 체크포인트, 옵티마이저 상태 및 약 1.29GB의 모델 압축파일은 용량과 재배포 문제로 Git에 포함하지 않습니다. 해당 파일은 로컬 `data/` 및 `outputs/` 아래에만 보관합니다.
