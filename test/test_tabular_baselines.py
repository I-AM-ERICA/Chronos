from __future__ import annotations

import json

import numpy as np
import pandas as pd

from chronos_sales.config import ExperimentConfig
from chronos_sales.tabular_baselines import (
    MAX_HISTORY,
    build_supervised_rows,
    run_tabular_benchmarks,
)


def test_supervised_features_use_only_earlier_observations():
    values = np.arange(100, dtype=np.float64)
    dates = pd.date_range("2020-01-01", periods=len(values), freq="D")
    features, targets, indices = build_supervised_rows(values, dates)
    assert indices[0] == MAX_HISTORY
    assert features.iloc[0]["lag_1"] == values[MAX_HISTORY - 1]
    assert features.iloc[0]["lag_56"] == values[0]
    assert targets[0] == values[MAX_HISTORY]


def test_random_forest_benchmark_keeps_holdouts_out_of_fit(tmp_path):
    total = 180
    dates = pd.date_range("2020-01-01", periods=total, freq="D")
    values = 100 + 10 * np.sin(np.arange(total) * 2 * np.pi / 7)
    prepared = tmp_path / "daily_revenue.parquet"
    pd.DataFrame(
        {
            "business_id": "ONLINE_RETAIL_UK",
            "date": dates,
            "net_revenue_gbp": values,
        }
    ).to_parquet(prepared, index=False)
    output = tmp_path / "outputs"
    config = ExperimentConfig(
        prediction_length=28,
        validation_length=28,
        num_samples=20,
    )
    comparison = run_tabular_benchmarks(
        prepared, output, config, model_names=["random_forest"]
    )
    audit = json.loads((output / "benchmark_audit.json").read_text())
    expected_train_length = total - 56
    assert audit["train_observations"] == expected_train_length
    assert (
        audit["models"]["random_forest"]["last_target_index"]
        == expected_train_length - 1
    )
    assert set(comparison["model"]) == {"random_forest"}
    assert set(pd.read_csv(output / "tabular_model_metrics.csv")["split"]) == {
        "validation",
        "test",
    }
