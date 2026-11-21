from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_revenue_prediction(
    prediction_path: Path, output_path: Path, store_id: str
) -> Path:
    payload = np.load(prediction_path, allow_pickle=False)
    samples = payload["samples"]
    actual = payload["actual"]
    ids = payload["series_ids"].astype(str)
    matches = np.flatnonzero(ids == store_id)
    if len(matches) != 1:
        raise KeyError(f"Store {store_id!r} was not found in predictions.")
    index = int(matches[0])
    days = np.arange(1, actual.shape[1] + 1)
    median = np.quantile(samples[index], 0.5, axis=0)
    low = np.quantile(samples[index], 0.1, axis=0)
    high = np.quantile(samples[index], 0.9, axis=0)
    plt.figure(figsize=(11, 5.5))
    plt.fill_between(days, low, high, color="#6ABF69", alpha=0.25, label="10–90% range")
    plt.plot(days, median, color="#137A4A", linewidth=2, label="Chronos median")
    plt.plot(days, actual[index], color="#222222", linewidth=2, label="Actual revenue proxy")
    plt.title(f"28-day revenue forecast — {store_id}")
    plt.xlabel("Forecast day")
    plt.ylabel("Revenue proxy")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
