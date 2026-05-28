import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.mdlm.mdlm_train_sft import TrainingConfig, run_training


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Convert Hydra config → TrainingConfig, then delegate to run_training."""
    override_dirname = HydraConfig.get().job.override_dirname or "default"

    training_cfg = TrainingConfig(
        # paths
        TRAIN_DATA_LOAD_PATH=cfg.TRAIN_DATA_LOAD_PATH,
        TEST_DATA_LOAD_PATH=cfg.TEST_DATA_LOAD_PATH,
        TRAIN_MODEL_LOAD_PATH=cfg.TRAIN_MODEL_LOAD_PATH,
        TRAIN_MODEL_SAVE_PATH=f"{cfg.TRAIN_MODEL_SAVE_PATH}/{override_dirname}",
        # dataset
        num_train_samples=cfg.num_train_samples,
        num_test_samples=cfg.num_test_samples,
        num_workers=cfg.num_workers,
        max_length=cfg.max_length,
        # training
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        time_epsilon=cfg.time_epsilon,
        # eval
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        report_to=list(cfg.report_to),
    )

    run_training(training_cfg)


if __name__ == "__main__":
    main()