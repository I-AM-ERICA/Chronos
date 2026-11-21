import numpy as np

from chronos_sales.config import ExperimentConfig
from chronos_sales.data import split_target, validation_indices


def test_last_28_days_never_enter_training_data():
    config = ExperimentConfig(prediction_length=28, validation_length=28)
    target = np.arange(200, dtype=np.float32)
    split = split_target(target, config)
    np.testing.assert_array_equal(split.train, target[:-56])
    np.testing.assert_array_equal(split.validation_truth, target[-56:-28])
    np.testing.assert_array_equal(split.test_context, target[:-28])
    np.testing.assert_array_equal(split.test_truth, target[-28:])
    assert set(split.train).isdisjoint(set(split.test_truth))


def test_split_returns_copies_not_views():
    config = ExperimentConfig()
    target = np.arange(100, dtype=np.float32)
    split = split_target(target, config)
    target[0] = -999
    assert split.train[0] == 0


def test_validation_sample_is_deterministic():
    first = validation_indices(100, 10, seed=42)
    second = validation_indices(100, 10, seed=42)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 10
