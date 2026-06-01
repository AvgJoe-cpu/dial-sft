import hydra
from omegaconf import DictConfig

from src.ar.ar_config_schema import (
    ExperimentConfig,
    TrainingConfig,
    InferenceConfig,
    register_configs,
)
from src.ar.ar_train_sft import run_training
from src.ar.ar_inference import run_inference

register_configs()


@hydra.main(config_path="../../conf", config_name="ar_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── TRAINING ─────────────────────────────────────────────────────────────
    print("[AR-SFT] Running training...")
    run_training(
        TrainingConfig(
            train_data_load_path=cfg.paths.train_data_load_path,
            train_model_load_path=cfg.paths.train_model_load_path,
            train_model_save_path=cfg.paths.train_model_save_path,
            num_samples=cfg.dataset.num_train_samples,
            num_epochs=cfg.training.num_epochs,
            batch_size=cfg.training.batch_size,
            logging_steps=cfg.training.logging_steps,
            optim=cfg.training.optim,
            bf16=cfg.training.bf16,
            use_liger_kernel=cfg.training.use_liger_kernel,
            dataloader_num_workers=cfg.training.dataloader_num_workers,
            dataloader_pin_memory=cfg.training.dataloader_pin_memory,
            assistant_only_loss=cfg.training.assistant_only_loss,
            report_to=cfg.training.report_to,
            push_to_hub=cfg.training.push_to_hub,
            remove_unused_columns=cfg.training.remove_unused_columns,
        )
    )
    print("✓ [AR-SFT] Training complete")

    # ── INFERENCE ───────────────────────────────────────────────────────────
    print("[AR-SFT] Running inference...")
    run_inference(
        InferenceConfig(
            infer_data_load_path=cfg.paths.infer_data_load_path,
            infer_data_save_path=cfg.paths.infer_data_save_path,
            infer_model_load_path=cfg.paths.infer_model_load_path,
            num_samples=cfg.dataset.num_infer_samples,
            max_new_tokens=cfg.inference.max_new_tokens,
            num_beams=cfg.inference.num_beams,
            do_sample=cfg.inference.do_sample,
            use_cache=cfg.inference.use_cache,
            temperature=cfg.inference.temperature,
            num_return_sequences=cfg.inference.num_return_sequences,
            batch_size=cfg.inference.batch_size,
        )
    )
    print("✓ [AR-SFT] Inference complete")


if __name__ == "__main__":
    main()
