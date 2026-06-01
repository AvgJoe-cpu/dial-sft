# (https://huggingface.co/docs/trl/sft_trainer$0)
# standard - conversational
# LM - prompt completion

from dataclasses import dataclass, field

from src.ar.ar_load_model import setup_model_and_tokenizer

import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk


@dataclass
class TrainingConfig:
    # paths
    TRAIN_DATA_LOAD_PATH: str = ""
    TRAIN_MODEL_LOAD_PATH: str = ""
    TRAIN_MODEL_SAVE_PATH: str = ""
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
