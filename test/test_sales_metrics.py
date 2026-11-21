import numpy as np

from chronos_sales.metrics import calculate_metrics


def test_perfect_forecast_has_zero_metrics():
    actual = np.array([[1.0, 2.0], [2.0, 4.0]])
    samples = np.repeat(actual[:, None, :], repeats=20, axis=1)
    contexts = [np.arange(1, 30), np.arange(2, 31)]
    metrics = calculate_metrics(samples, actual, contexts, seasonal_period=7)
    assert metrics["WQL"] == 0.0
    assert metrics["MASE"] == 0.0
    assert metrics["MAE"] == 0.0


def test_metrics_reject_wrong_shape():
    try:
        calculate_metrics(np.ones((3, 4)), np.ones((3, 4)), [np.ones(10)] * 3)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("Expected ValueError")
