from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import SplitName, evaluation_arrays
from .metrics import calculate_metrics


def resolve_device(device: str) -> tuple[str, torch.dtype]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype


def predict_samples(
    model_id: str,
    contexts: list[np.ndarray],
    config: ExperimentConfig,
    device: str = "auto",
    batch_size: int = 32,
) -> np.ndarray:
    from chronos import ChronosPipeline

    device, dtype = resolve_device(device)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    pipeline = ChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )
    outputs = []
    for start in tqdm(range(0, len(contexts), batch_size), desc="forecast"):
        batch = [torch.tensor(x, dtype=torch.float32) for x in contexts[start : start + batch_size]]
        result = pipeline.predict(
            batch,
            prediction_length=config.prediction_length,
            num_samples=config.num_samples,
        )
        outputs.append(result.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def run_evaluation(
    model_id: str,
    output_dir: Path,
    config: ExperimentConfig,
    split_name: SplitName = "test",
    indices: np.ndarray | None = None,
    device: str = "auto",
    batch_size: int = 32,
    label: str = "model",
    save_predictions: bool = True,
    dataset=None,
    dataset_name: str = "online-retail-revenue",
) -> dict[str, float | int | str]:
    if dataset is None:
        raise ValueError("dataset is required for revenue evaluation")
    contexts, actual, series_ids = evaluation_arrays(
        dataset, config, split_name=split_name, indices=indices
    )
    samples = predict_samples(model_id, contexts, config, device, batch_size)
    metrics = calculate_metrics(samples, actual, contexts, config.seasonal_period)
    row: dict[str, float | int | str] = {
        "dataset": dataset_name,
        "split": split_name,
        "model": model_id,
        "label": label,
        "series_count": len(contexts),
        **metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        np.savez_compressed(
            output_dir / f"{label}_{split_name}_predictions.npz",
            samples=samples,
            actual=actual.astype(np.float32),
            series_ids=np.asarray(series_ids),
        )
    return row


def save_metric_row(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
