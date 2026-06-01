# (https://huggingface.co/docs/trl/sft_trainer$0)
# standard - conversational
# LM - prompt completion

from src.ar.ar_load_model import setup_model_and_tokenizer
from src.ar.ar_inference import run_inference

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


def run_training(
    TRAIN_DATA_LOAD_PATH: str,
    TRAIN_MODEL_LOAD_PATH: str,
    TRAIN_MODEL_SAVE_PATH: str,
    num_samples: int = 10000,
    num_epochs: int = 4,
    batch_size: int = 64,
    # is_local: bool = False,
):
    output_dir = TRAIN_MODEL_SAVE_PATH
    model_name = TRAIN_MODEL_LOAD_PATH
    dataset_path = TRAIN_DATA_LOAD_PATH

    train_ds = load_from_disk(dataset_path)

    print(f"Selecting {num_samples} samples and preprocessing...")
    train_ds = train_ds.select(range(num_samples)).rename_column("story", "completion")

    train_dataset = train_ds.map(format_to_messages)
    #del train_ds

    print(f"Loading model: {model_name}")
    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_name, for_training=True
    )

    training_args = SFTConfig(
        push_to_hub=False,
        output_dir=output_dir,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/tb_logs",
        bf16=True,
        optim="adamw_torch_fused",
        use_liger_kernel=False,  # False on mps
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=10,
        assistant_only_loss=True,  # Loss ONLY on assistant messages
        remove_unused_columns=False,  # Important to keep all columns for the processing class
    )

    print(f"Starting training with {num_samples} samples...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    print(f"✓ Training complete. Model saved to: {output_dir}")
    torch.cuda.empty_cache()
    del model, tokenizer, training_args, trainer, train_dataset




if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        TRAIN_OG_PATH = "euclaise/writingprompts"
        TRAIN_OG_SAVE_PATH = f"{tmp}/train_ds_og"
        EVAL_OG_SAVE_PATH = f"{tmp}/eval_ds_og"
        TEST_OG_SAVE_PATH = f"{tmp}/test_ds_og"
        # ── PREPARE DATASETS ───────────────────────────────────────────────

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

            train_data_load_path = config["TRAIN_DATA_LOAD_PATH"]
            train_model_load_path = config["TRAIN_MODEL_LOAD_PATH"]
            train_model_save_path = config["TRAIN_MODEL_SAVE_PATH"]
            infer_model_load_path = config["INFER_MODEL_LOAD_PATH"]
            infer_data_load_path = config["INFER_DATA_LOAD_PATH"]
            infer_data_save_path = config["INFER_DATA_SAVE_PATH"]

            print(f"Training data:        {train_data_load_path}")
            print(f"Training model:       {train_model_load_path}")
            print(f"Will save model to:   {train_model_save_path}")

            # ── TRAINING ─────────────────────────────────────────────────────
            print(f"[{round_name}] Running training...")
            run_training(
                TRAIN_DATA_LOAD_PATH=train_data_load_path,
                TRAIN_MODEL_LOAD_PATH=train_model_load_path,
                TRAIN_MODEL_SAVE_PATH=train_model_save_path,
                num_samples=100,
            )
            print(f"✓ [{round_name}] Training complete\n")

            # ── INFERENCE ────────────────────────────────────────────────────
            print(f"[{round_name}] Running inference...")
            run_inference(
                INFER_MODEL_LOAD_PATH=infer_model_load_path,
                INFER_DATA_LOAD_PATH=infer_data_load_path,
                INFER_DATA_SAVE_PATH=infer_data_save_path,
                num_samples=100,
            )
            print(f"✓ [{round_name}] Inference complete\n")

            print(f"{'='*80}")
            print(f"✓ {round_name} FINISHED")
            print(f"{'='*80}\n")
