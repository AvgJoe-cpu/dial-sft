# (https://huggingface.co/docs/trl/sft_trainer$0)
# standard - conversational
# LM - prompt completion

from src.ar.ar_load_model import setup_model_and_tokenizer
from src.ar.ar_inference import run_inference
from src.ar.ar_config_schema import TrainingConfig, InferenceConfig

import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk


def format_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


def run_training(cfg: TrainingConfig) -> None:
    train_ds = load_from_disk(cfg.TRAIN_DATA_LOAD_PATH)

    print(f"Selecting {cfg.num_samples} samples and preprocessing...")
    train_ds = train_ds.select(range(cfg.num_samples)).rename_column("story", "completion")
    train_dataset = train_ds.map(format_to_messages)

    print(f"Loading model: {cfg.TRAIN_MODEL_LOAD_PATH}")
    model, tokenizer = setup_model_and_tokenizer(
        model_name=cfg.TRAIN_MODEL_LOAD_PATH, for_training=True
    )

    training_args = SFTConfig(
        push_to_hub=cfg.push_to_hub,
        output_dir=cfg.TRAIN_MODEL_SAVE_PATH,
        report_to=cfg.report_to,
        logging_dir=f"{cfg.TRAIN_MODEL_SAVE_PATH}/tb_logs",
        bf16=cfg.bf16,
        optim=cfg.optim,
        use_liger_kernel=cfg.use_liger_kernel,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        logging_steps=cfg.logging_steps,
        assistant_only_loss=cfg.assistant_only_loss,
        remove_unused_columns=cfg.remove_unused_columns,
    )

    print(f"Starting training with {cfg.num_samples} samples...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    print(f"✓ Training complete. Model saved to: {cfg.TRAIN_MODEL_SAVE_PATH}")
    torch.cuda.empty_cache()
    del model, tokenizer, training_args, trainer, train_dataset


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        TRAIN_OG_SAVE_PATH = f"{tmp}/train_ds_og"

        CONFIG_DICT = {
            "ROUND1": {
                "TRAIN_DATA_LOAD_PATH": TRAIN_OG_SAVE_PATH,
                "TRAIN_MODEL_LOAD_PATH": "EleutherAI/pythia-70m",
                "TRAIN_MODEL_SAVE_PATH": f"{tmp}/sft_output",
                "INFER_MODEL_LOAD_PATH": f"{tmp}/sft_output",
                "INFER_DATA_LOAD_PATH": TRAIN_OG_SAVE_PATH,
                "INFER_DATA_SAVE_PATH": f"{tmp}/local_arrow_dataset",
            },
        }

        for round_name, config in CONFIG_DICT.items():
            print(f"\n{'='*80}")
            print(f"STARTING {round_name}")
            print(f"{'='*80}\n")

            print(f"Training data:        {config['TRAIN_DATA_LOAD_PATH']}")
            print(f"Training model:       {config['TRAIN_MODEL_LOAD_PATH']}")
            print(f"Will save model to:   {config['TRAIN_MODEL_SAVE_PATH']}")

            # ── TRAINING ─────────────────────────────────────────────────────
            print(f"[{round_name}] Running training...")
            run_training(TrainingConfig(
                TRAIN_DATA_LOAD_PATH=config["TRAIN_DATA_LOAD_PATH"],
                TRAIN_MODEL_LOAD_PATH=config["TRAIN_MODEL_LOAD_PATH"],
                TRAIN_MODEL_SAVE_PATH=config["TRAIN_MODEL_SAVE_PATH"],
                num_samples=100,
            ))
            print(f"✓ [{round_name}] Training complete\n")

            # ── INFERENCE ────────────────────────────────────────────────────
            print(f"[{round_name}] Running inference...")
            run_inference(InferenceConfig(
                INFER_MODEL_LOAD_PATH=config["INFER_MODEL_LOAD_PATH"],
                INFER_DATA_LOAD_PATH=config["INFER_DATA_LOAD_PATH"],
                INFER_DATA_SAVE_PATH=config["INFER_DATA_SAVE_PATH"],
                num_samples=100,
            ))
            print(f"✓ [{round_name}] Inference complete\n")

            print(f"{'='*80}")
            print(f"✓ {round_name} FINISHED")
            print(f"{'='*80}\n")
