from __future__ import annotations

import numpy as np


QUANTILES = np.arange(0.1, 1.0, 0.1)


def calculate_quantile_metrics(
    quantile_forecasts: np.ndarray,
    actual: np.ndarray,
    contexts: list[np.ndarray],
    seasonal_period: int = 7,
) -> dict[str, float]:
    """Compute metrics from q10..q90 forecasts.

    quantile_forecasts must have shape [series, 9, horizon] and follow the
    order in ``QUANTILES``. This common path lets sample-based and native
    probabilistic models use exactly the same metric implementation.
    """
    forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if forecasts.ndim != 3 or forecasts.shape[1] != len(QUANTILES):
        raise ValueError("quantile_forecasts must have shape [series, 9, horizon]")
    if actual.shape != (forecasts.shape[0], forecasts.shape[2]):
        raise ValueError("actual shape must match forecasts' series and horizon axes")
    if len(contexts) != forecasts.shape[0]:
        raise ValueError("contexts length must match forecasts' series axis")

    median = forecasts[:, 4, :]
    mae = float(np.mean(np.abs(actual - median)))
    mase_rows = []
    for forecast, truth, context in zip(median, actual, contexts):
        history = np.asarray(context, dtype=np.float64)
        if len(history) <= seasonal_period:
            scale = np.mean(np.abs(np.diff(history)))
        else:
            scale = np.mean(np.abs(history[seasonal_period:] - history[:-seasonal_period]))
        if np.isfinite(scale) and scale > 0:
            mase_rows.append(np.mean(np.abs(truth - forecast)) / scale)
    mase = float(np.mean(mase_rows)) if mase_rows else float("nan")

    denominator = float(np.sum(np.abs(actual)))
    quantile_losses = []
    for quantile_index, quantile in enumerate(QUANTILES):
        error = actual - forecasts[:, quantile_index, :]
        pinball = np.maximum(quantile * error, (quantile - 1.0) * error)
        quantile_losses.append(
            2.0 * float(np.sum(pinball)) / denominator
            if denominator > 0
            else float("nan")
        )
    wql = float(np.mean(quantile_losses))
    wape = (
        float(np.sum(np.abs(actual - median))) / denominator
        if denominator > 0
        else float("nan")
    )
    return {"WQL": wql, "MASE": mase, "MAE": mae, "WAPE": wape}


def calculate_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    contexts: list[np.ndarray],
    seasonal_period: int = 7,
) -> dict[str, float]:
    """Compute aggregate WQL, mean per-series MASE, and reference MAE."""
    samples = np.asarray(samples, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("samples must have shape [series, samples, horizon]")
    if actual.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("actual shape must match samples' series and horizon axes")

    forecasts = np.quantile(samples, QUANTILES, axis=1).transpose(1, 0, 2)
    return calculate_quantile_metrics(forecasts, actual, contexts, seasonal_period)
