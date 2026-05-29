"""
Hydra structured config schema for dial-sft.

Registered with ConfigStore so Hydra validates every launch against
these types before any model or data is loaded. No GPU time is wasted
on misconfigured experiments.
"""

from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = "mdlm-0.2b"
    scheduler: str = MISSING  # "linear" | "cosine"
    loss_weight_type: str = MISSING  # "uniform" | "scheduler"
    time_epsilon: float = 0.001


@dataclass
class PathsConfig:
    train_data: str = MISSING
    test_data: str = MISSING
    model_load: str = MISSING
    model_save: str = MISSING
    log_dir: str = MISSING


@dataclass
class DatasetConfig:
    num_workers: int = 4
    max_length: int = 1024
    id_field: str = "id"
    stage: int = 0
    num_train_samples: int = -1
    num_test_samples: int = -1

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class TrainingConfig:
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    logging_steps: int = 50
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999


@dataclass
class EvalConfig:
    strategy: str = "steps"
    steps: int = 100
    save_strategy: str = "no"
    metric: str = "masked_lm_loss"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class ExperimentConfig:
    model: ModelConfig = MISSING
    paths: PathsConfig = MISSING
    dataset: DatasetConfig = MISSING
    training: TrainingConfig = MISSING
    eval: EvalConfig = MISSING
    seed: int = 42
    stage_name: str = MISSING


def register_configs() -> None:
    """Register the structured config schema with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ExperimentConfig)
