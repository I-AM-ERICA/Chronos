from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import split_target


REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}


class OnlineRetailDataError(ValueError):
    """Raised when Online Retail II cannot produce a trustworthy daily series."""


def _read_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Online Retail II workbook was not found: {path}")
    sheets = pd.read_excel(path, sheet_name=None)
    if not sheets:
        raise OnlineRetailDataError("The workbook contains no worksheets.")
    frames = []
    for sheet_name, frame in sheets.items():
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise OnlineRetailDataError(
                f"Sheet {sheet_name!r} is missing columns: {sorted(missing)}"
            )
        selected = frame[list(REQUIRED_COLUMNS)].copy()
        selected["source_sheet"] = sheet_name
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def prepare_online_retail_revenue(
    source_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> dict:
    """Convert Online Retail II transactions into one continuous daily revenue series."""
    from gluonts.dataset.arrow import ArrowWriter

    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_source(source_path)
    raw_line_count = len(raw)

    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    required_null = raw[["Invoice", "StockCode", "InvoiceDate", "Quantity", "Price"]].isna()
    if required_null.any().any():
        counts = required_null.sum()
        raise OnlineRetailDataError(
            "Required transaction values are missing: "
            + ", ".join(f"{key}={int(value)}" for key, value in counts.items() if value)
        )

    dedup_columns = list(REQUIRED_COLUMNS)
    duplicate_mask = raw.duplicated(dedup_columns, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    transactions = raw.loc[~duplicate_mask].copy()

    transactions["invoice_id"] = transactions["Invoice"].astype(str)
    transactions["is_cancellation"] = transactions["invoice_id"].str.upper().str.startswith("C")
    transactions["date"] = transactions["InvoiceDate"].dt.normalize()
    transactions["line_revenue_gbp"] = (
        transactions["Quantity"].astype(np.float64)
        * transactions["Price"].astype(np.float64)
    )

    negative_price = transactions["Price"] < 0
    negative_price_rows = transactions.loc[
        negative_price,
        [
            "source_sheet",
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Country",
            "line_revenue_gbp",
        ],
    ].copy()
    negative_price_rows["issue"] = "negative_price_excluded"
    negative_price_rows.to_csv(
        output_dir / "excluded_negative_price_rows.csv", index=False
    )
    transactions = transactions.loc[~negative_price].copy()

    transactions["gross_component"] = np.where(
        transactions["Quantity"] > 0, transactions["line_revenue_gbp"], 0.0
    )
    transactions["returns_component"] = np.where(
        transactions["Quantity"] < 0, transactions["line_revenue_gbp"], 0.0
    )
    daily_observed = transactions.groupby("date", as_index=True).agg(
        gross_revenue_gbp=("gross_component", "sum"),
        returns_revenue_gbp=("returns_component", "sum"),
        net_revenue_gbp=("line_revenue_gbp", "sum"),
        transaction_lines=("Invoice", "size"),
        cancellation_lines=("is_cancellation", "sum"),
    )
    if daily_observed.empty:
        raise OnlineRetailDataError("No valid transactions remain after cleaning.")

    all_dates = pd.date_range(daily_observed.index.min(), daily_observed.index.max(), freq="D")
    daily = daily_observed.reindex(all_dates, fill_value=0.0)
    daily.index.name = "date"
    daily = daily.reset_index()
    daily.insert(0, "business_id", "ONLINE_RETAIL_UK")
    daily["transaction_lines"] = daily["transaction_lines"].astype(np.int64)
    daily["cancellation_lines"] = daily["cancellation_lines"].astype(np.int64)

    reconciliation_error = np.abs(
        daily["gross_revenue_gbp"]
        + daily["returns_revenue_gbp"]
        - daily["net_revenue_gbp"]
    ).max()
    if not np.isfinite(reconciliation_error) or reconciliation_error > 1e-6:
        raise OnlineRetailDataError(
            f"Daily revenue reconciliation failed: max_error={reconciliation_error}"
        )
    daily.to_parquet(output_dir / "daily_revenue.parquet", index=False)
    daily.to_csv(output_dir / "daily_revenue.csv", index=False)

    dataset = load_online_retail_revenue(output_dir / "daily_revenue.parquet")
    row = dataset[0]
    split = split_target(row["target"], config)
    if len(split.train) < config.min_past + config.prediction_length:
        raise OnlineRetailDataError(
            "The training prefix is too short for the configured sampling window."
        )
    arrow_dir = output_dir / "arrow" / "train"
    arrow_dir.mkdir(parents=True, exist_ok=True)
    ArrowWriter(compression="lz4").write_to_file(
        iter(
            [
                {
                    "start": np.datetime64(row["timestamp"][0]),
                    "target": split.train.astype(np.float32),
                }
            ]
        ),
        arrow_dir / "data.arrow",
    )

    audit = pd.DataFrame(
        [
            {
                "raw_rows": raw_line_count,
                "exact_duplicates_removed": duplicate_count,
                "negative_price_rows_excluded": int(negative_price.sum()),
                "valid_transaction_rows": len(transactions),
                "cancellation_rows": int(transactions["is_cancellation"].sum()),
                "negative_quantity_rows": int((transactions["Quantity"] < 0).sum()),
                "zero_price_rows": int((transactions["Price"] == 0).sum()),
                "start_date": daily["date"].min().date().isoformat(),
                "end_date": daily["date"].max().date().isoformat(),
                "calendar_days": len(daily),
                "zero_revenue_days": int((daily["net_revenue_gbp"] == 0).sum()),
                "reconciliation_max_error": float(reconciliation_error),
            }
        ]
    )
    audit.to_csv(output_dir / "transaction_audit.csv", index=False)

    timestamps = pd.DatetimeIndex(row["timestamp"])
    manifest_row = {
        "business_id": row["id"],
        "total_length": len(row["target"]),
        "start_date": timestamps[0].date().isoformat(),
        "train_end_date": timestamps[len(split.train) - 1].date().isoformat(),
        "validation_start_date": timestamps[len(split.train)].date().isoformat(),
        "test_start_date": timestamps[-config.prediction_length].date().isoformat(),
        "test_end_date": timestamps[-1].date().isoformat(),
        "train_length": len(split.train),
    }
    pd.DataFrame([manifest_row]).to_csv(
        output_dir / "online_retail_manifest.csv", index=False
    )
    manifest = {
        "source": "UCI Machine Learning Repository: Online Retail II",
        "source_url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "license": "CC BY 4.0",
        "citation": "Chen, D. (2012). Online Retail II. DOI: 10.24432/C5CG6D",
        "target": "net_product_revenue_proxy_gbp",
        "target_definition": "daily sum(Quantity * Price), including returns; negative prices excluded",
        "not_accounting_net_revenue": True,
        "series_count": 1,
        "day_count": len(daily),
        "split_policy": {
            "train": "target[:-56]",
            "validation": "target[-56:-28]",
            "test": "target[-28:]",
        },
        "config": asdict(config),
        "arrow_path": str(arrow_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_online_retail_revenue(path: Path) -> list[dict]:
    frame = pd.read_parquet(path)
    required = {"business_id", "date", "net_revenue_gbp"}
    missing = required - set(frame.columns)
    if missing:
        raise OnlineRetailDataError(f"Prepared revenue data is missing: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date")
    dates = pd.DatetimeIndex(frame["date"])
    if dates.duplicated().any() or not np.all(
        np.diff(dates.values) == np.timedelta64(1, "D")
    ):
        raise OnlineRetailDataError("Prepared dates must be unique and daily.")
    target = frame["net_revenue_gbp"].to_numpy(dtype=np.float32)
    if not np.isfinite(target).all():
        raise OnlineRetailDataError("Prepared revenue must be finite.")
    business_ids = frame["business_id"].astype(str).unique()
    if len(business_ids) != 1:
        raise OnlineRetailDataError("Online Retail II must contain one business series.")
    return [
        {
            "id": str(business_ids[0]),
            "timestamp": dates.to_pydatetime().tolist(),
            "target": target,
        }
    ]
