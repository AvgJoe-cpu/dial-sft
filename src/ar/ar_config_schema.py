from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class PathsConfig:
    train_data_load_path: str = MISSING
    train_model_load_path: str = MISSING
    train_model_save_path: str = MISSING
    infer_data_load_path: str = MISSING
    infer_model_load_path: str = MISSING
    infer_data_save_path: str = MISSING


@dataclass
class DatasetConfig:
    num_train_samples: int = 10000
    num_infer_samples: int = 10000


@dataclass
class TrainingConfig:
    # paths
    train_data_load_path: str = ""
    train_model_load_path: str = ""
    train_model_save_path: str = ""
    # dataset
    num_samples: int = 10000
    # training
    num_epochs: int = 4
    batch_size: int = 64
    logging_steps: int = 10
    # optimizer
    optim: str = "adamw_torch_fused"
    # precision / hardware
    bf16: bool = True
    use_liger_kernel: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    # loss
    assistant_only_loss: bool = True
    # misc
    report_to: str = "tensorboard"
    push_to_hub: bool = False
    remove_unused_columns: bool = False


@dataclass
class InferenceConfig:
    # paths
    infer_model_load_path: str = ""
    infer_data_load_path: str = ""
    infer_data_save_path: str = ""
    # dataset
    num_samples: int = 10000
    # generation
    max_new_tokens: int = 512
    num_beams: int = 1
    do_sample: bool = True
    use_cache: bool = True
    temperature: float = 1.1
    num_return_sequences: int = 1
    batch_size: int = 10


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExperimentConfig)
