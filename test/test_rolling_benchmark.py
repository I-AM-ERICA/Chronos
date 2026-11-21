from __future__ import annotations

import json

import numpy as np
import pandas as pd

from chronos_sales.config import ExperimentConfig
from chronos_sales.metrics import QUANTILES, calculate_metrics, calculate_quantile_metrics
from chronos_sales.rolling_benchmark import (
    RANKED_MODELS,
    _stat_quantiles,
    make_fold_specs,
    summarize_rolling_benchmark,
)


def test_three_fold_dates_match_online_retail_plan():
    dates = pd.date_range("2009-12-01", periods=739, freq="D")
    specs = make_fold_specs(dates, ExperimentConfig(), folds=3)
    assert [spec.train_end for spec in specs] == [627, 655, 683]
    assert [spec.train_end_date for spec in specs] == [
        "2011-08-19",
        "2011-09-16",
        "2011-10-14",
    ]
    assert [spec.test_start_date for spec in specs] == [
        "2011-09-17",
        "2011-10-15",
        "2011-11-12",
    ]
    assert specs[-1].test_end_date == "2011-12-09"


def test_sample_and_quantile_metric_paths_are_identical():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(1, 20, 28))
    actual = rng.normal(size=(1, 28))
    contexts = [rng.normal(size=100)]
    from_samples = calculate_metrics(samples, actual, contexts)
    quantiles = np.quantile(samples, QUANTILES, axis=1).transpose(1, 0, 2)
    direct = calculate_quantile_metrics(quantiles, actual, contexts)
    assert from_samples == direct


def test_statistical_interval_keys_map_to_q10_through_q90():
    result = {
        "lo-80": np.array([1.0]),
        "lo-60": np.array([2.0]),
        "lo-40": np.array([3.0]),
        "lo-20": np.array([4.0]),
        "mean": np.array([5.0]),
        "hi-20": np.array([6.0]),
        "hi-40": np.array([7.0]),
        "hi-60": np.array([8.0]),
        "hi-80": np.array([9.0]),
    }
    assert _stat_quantiles(result)[:, 0].tolist() == list(range(1, 10))


def test_summary_ranks_mean_wape_and_writes_decision(tmp_path):
    folds = []
    for fold in range(1, 4):
        fold_dir = tmp_path / "folds" / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True)
        spec = {
            "fold": fold,
            "train_end": 100 + fold,
            "validation_start": 100 + fold,
            "validation_end": 128 + fold,
            "test_start": 128 + fold,
            "test_end": 156 + fold,
            "train_end_date": f"2020-01-0{fold}",
            "validation_start_date": f"2020-02-0{fold}",
            "validation_end_date": f"2020-02-2{fold}",
            "test_start_date": f"2020-03-0{fold}",
            "test_end_date": f"2020-03-2{fold}",
        }
        folds.append(spec)
        rows = []
        for index, model in enumerate(RANKED_MODELS):
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "split": "test",
                    "WQL": 0.1 + index * 0.01,
                    "MASE": 0.8 + index * 0.01,
                    "MAE": 100 + index,
                    "WAPE": 0.1 + index * 0.01,
                }
            )
        pd.DataFrame(rows[:1]).to_csv(fold_dir / "chronos_metrics.csv", index=False)
        pd.DataFrame(rows[1:]).to_csv(fold_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(tmp_path / "folds.csv", index=False)
    ranking = summarize_rolling_benchmark(tmp_path)
    assert ranking.iloc[0]["model"] == "chronos_finetuned"
    decision = json.loads((tmp_path / "validity_decision.json").read_text())
    assert decision["pass"] is True
