from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.common import FileDataset

from chronos_sales.config import ExperimentConfig
from chronos_sales.online_retail import (
    load_online_retail_revenue,
    prepare_online_retail_revenue,
)


def make_online_retail_workbook(path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    rows = []
    for index, date in enumerate(dates):
        if index == 30:
            continue
        rows.append(
            {
                "Invoice": f"I{index:04d}",
                "StockCode": "SKU-1",
                "Description": "Product",
                "Quantity": 2,
                "InvoiceDate": date + pd.Timedelta(hours=10),
                "Price": 10.0,
                "Customer ID": 1,
                "Country": "United Kingdom",
            }
        )
    rows.append(
        {
            "Invoice": "C0060",
            "StockCode": "SKU-1",
            "Description": "Product return",
            "Quantity": -1,
            "InvoiceDate": dates[60] + pd.Timedelta(hours=11),
            "Price": 5.0,
            "Customer ID": 1,
            "Country": "United Kingdom",
        }
    )
    rows.append(
        {
            "Invoice": "ADJUST",
            "StockCode": "ADJ",
            "Description": "Adjustment",
            "Quantity": 1,
            "InvoiceDate": dates[20],
            "Price": -2.0,
            "Customer ID": np.nan,
            "Country": "United Kingdom",
        }
    )
    frame = pd.DataFrame(rows)
    first = frame[frame["InvoiceDate"].dt.normalize() <= dates[49]].copy()
    second = frame[frame["InvoiceDate"].dt.normalize() >= dates[40]].copy()
    first = pd.concat([first, first.iloc[[0]]], ignore_index=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        first.to_excel(writer, sheet_name="Year 1", index=False)
        second.to_excel(writer, sheet_name="Year 2", index=False)
    return path


def test_online_retail_preparation_deduplicates_and_reconciles(tmp_path):
    source = make_online_retail_workbook(tmp_path / "online_retail_II.xlsx")
    output = tmp_path / "prepared"
    config = ExperimentConfig(min_past=1)
    manifest = prepare_online_retail_revenue(source, output, config)

    daily = pd.read_parquet(output / "daily_revenue.parquet")
    assert manifest["license"] == "CC BY 4.0"
    assert len(daily) == 90
    assert daily.loc[daily["date"] == pd.Timestamp("2020-01-31"), "net_revenue_gbp"].item() == 0
    day_60 = daily.loc[daily["date"] == pd.Timestamp("2020-03-01")].iloc[0]
    assert day_60["gross_revenue_gbp"] == 20
    assert day_60["returns_revenue_gbp"] == -5
    assert day_60["net_revenue_gbp"] == 15
    assert np.allclose(
        daily["gross_revenue_gbp"] + daily["returns_revenue_gbp"],
        daily["net_revenue_gbp"],
    )

    audit = pd.read_csv(output / "transaction_audit.csv").iloc[0]
    assert audit["exact_duplicates_removed"] == 11
    assert audit["negative_price_rows_excluded"] == 1
    excluded = pd.read_csv(output / "excluded_negative_price_rows.csv")
    assert set(excluded["issue"]) == {"negative_price_excluded"}

    arrow_rows = list(FileDataset(output / "arrow" / "train", freq="D"))
    assert len(arrow_rows) == 1
    assert len(arrow_rows[0]["target"]) == 34
    dataset = load_online_retail_revenue(output / "daily_revenue.parquet")
    assert dataset[0]["id"] == "ONLINE_RETAIL_UK"
    assert len(dataset[0]["target"]) == 90
