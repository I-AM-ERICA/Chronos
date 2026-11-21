from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .config import ExperimentConfig


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SeriesSplit:
    train: np.ndarray
    validation_context: np.ndarray
    validation_truth: np.ndarray
    test_context: np.ndarray
    test_truth: np.ndarray


def split_target(target: Iterable[float], config: ExperimentConfig) -> SeriesSplit:
    values = np.asarray(target, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional target, got {values.ndim}D")
    if len(values) <= config.holdout_length:
        raise ValueError(
            f"Series length {len(values)} must be greater than holdout "
            f"length {config.holdout_length}."
        )
    validation_start = len(values) - config.holdout_length
    test_start = len(values) - config.prediction_length
    return SeriesSplit(
        train=values[:validation_start].copy(),
        validation_context=values[:validation_start].copy(),
        validation_truth=values[validation_start:test_start].copy(),
        test_context=values[:test_start].copy(),
        test_truth=values[test_start:].copy(),
    )


def _series_id(row: dict, index: int) -> str:
    for key in ("id", "item_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return f"series_{index:05d}"


def evaluation_arrays(
    dataset,
    config: ExperimentConfig,
    split_name: SplitName,
    indices: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    if split_name == "train":
        raise ValueError("Training split has no forecast truth window.")
    if indices is None:
        indices = np.arange(len(dataset))
    contexts: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    ids: list[str] = []
    for index in indices.tolist():
        row = dataset[int(index)]
        split = split_target(row["target"], config)
        if split_name == "validation":
            contexts.append(split.validation_context)
            truths.append(split.validation_truth)
        else:
            contexts.append(split.test_context)
            truths.append(split.test_truth)
        ids.append(_series_id(row, int(index)))
    return contexts, np.stack(truths), ids


def validation_indices(total: int, requested: int, seed: int) -> np.ndarray:
    count = min(total, requested)
    return np.sort(np.random.default_rng(seed).choice(total, count, replace=False))
