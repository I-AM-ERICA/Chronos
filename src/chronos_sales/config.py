from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "amazon/chronos-t5-small"
    dataset_repo: str = "uci-online-retail-ii"
    dataset_name: str = "online-retail-revenue"
    frequency: str = "D"
    context_length: int = 512
    prediction_length: int = 28
    validation_length: int = 28
    seasonal_period: int = 7
    min_past: int = 60
    num_samples: int = 20
    seed: int = 42
    validation_series: int = 1
    max_steps: int = 1000
    save_steps: int = 100
    learning_rate: float = 0.001
    effective_batch_size: int = 32
    target_name: str = "net_product_revenue_proxy_gbp"
    currency: str = "GBP"
    monte_carlo_draws: int = 10_000

    @property
    def holdout_length(self) -> int:
        return self.validation_length + self.prediction_length


def load_config(path: Path | str | None = None) -> ExperimentConfig:
    if path is None:
        path = PROJECT_ROOT / "configs" / "online-retail-revenue.yaml"
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    allowed = ExperimentConfig.__dataclass_fields__.keys()
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    return ExperimentConfig(**raw)
