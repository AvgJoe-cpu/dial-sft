import hydra
from hydra.core.hydra_config import HydraConfig

from src.config_schema import ExperimentConfig, register_configs
from src.mdlm.mdlm_train_sft import TrainingConfig, run_training
from src.paths import PathResolver

register_configs()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig) -> None:
    """Convert Hydra config → TrainingConfig, then delegate to run_training."""
    override_dirname = HydraConfig.get().job.override_dirname or "default"
    resolved_paths = PathResolver().resolve_mdlm_paths(cfg.paths, override_dirname)

    training_cfg = TrainingConfig(
        # paths
        TRAIN_DATA_LOAD_PATH=str(resolved_paths.train_data),
        TEST_DATA_LOAD_PATH=str(resolved_paths.test_data),
        TRAIN_MODEL_LOAD_PATH=str(resolved_paths.model_load),
        TRAIN_MODEL_SAVE_PATH=str(resolved_paths.model_save),
        # dataset
        num_train_samples=cfg.dataset.get("num_train_samples", -1),
        num_test_samples=cfg.dataset.get("num_test_samples", -1),
        num_workers=cfg.dataset.num_workers,
        max_length=cfg.dataset.max_length,
        # training
        num_epochs=cfg.training.num_epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        logging_steps=cfg.training.logging_steps,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        seed=cfg.seed,
        # model (MDLM)
        scheduler=cfg.model.scheduler,
        loss_weight_type=cfg.model.loss_weight_type,
        time_epsilon=cfg.model.time_epsilon,
        # eval
        eval_strategy=cfg.eval.strategy,
        eval_steps=cfg.eval.steps,
        save_strategy=cfg.eval.save_strategy,
        report_to=list(cfg.eval.report_to),
    )

    run_training(training_cfg)


if __name__ == "__main__":
    main()
