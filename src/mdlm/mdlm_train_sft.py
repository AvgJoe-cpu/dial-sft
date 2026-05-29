import gc
from dataclasses import dataclass, field

import torch

from datasets import load_from_disk
from src.mdlm.load_model import load_model
from src.mdlm.mdlm_helpers.mdlm_scheduler import (BaseAlphaScheduler,
                                                  LinearAlphaScheduler, make_alpha_scheduler)
from src.mdlm.mdlm_helpers.mdlm_trainer_sft import (MDLMConfig, MDLMSFTTrainer,
                                                    NLLPPLMetricComputer,
                                                    SFTCollator)


@dataclass
class TrainingConfig:
    # --- data & model paths ---
    TRAIN_DATA_LOAD_PATH: str = "./datasets/base/writingprompts_train"
    TEST_DATA_LOAD_PATH: str = "./datasets/base/writingprompts_test"
    TRAIN_MODEL_LOAD_PATH: str = "./weights/base"
    TRAIN_MODEL_SAVE_PATH: str = "./weights/checkpoints"

    # --- dataset ---
    num_train_samples: int = 1000
    num_test_samples: int = 1000

    num_workers: int = 4
    max_length: int = 512

    # --- MDLMConfig / TrainingArguments contract ---
    num_epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03,
    logging_steps: int = 1
    scheduler: str = "linear"            # "linear" | "cosine"
    loss_weight_type: str = "uniform"    # "uniform" | "scheduler"
    time_epsilon: float = 0.001

    # --- reporting & checkpointing ---
    eval_strategy: str = "steps"
    eval_steps: int = 50    
    save_strategy: str = "no"
    report_to: list = field(default_factory=lambda: ["tensorboard"])

def run_training(config: TrainingConfig = TrainingConfig()):
    def format_to_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
        }

    def _sft_map_fn(example, max_length=config.max_length):
        enc = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            max_length=max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
        labels = [tok if m == 1 else -100 for tok, m in zip(input_ids, assistant_mask)]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "assistant_mask": assistant_mask,
        }

    model, tokenizer = load_model(local_dir=config.TRAIN_MODEL_LOAD_PATH)

    train_ds = load_from_disk(config.TRAIN_DATA_LOAD_PATH)
    if config.num_train_samples and config.num_train_samples > 0:
        train_ds = train_ds.select(range(min(config.num_train_samples, len(train_ds))))

    train_ds = train_ds.map(format_to_messages).map(_sft_map_fn)
    train_ds = train_ds.select_columns(["input_ids", "labels", "assistant_mask"])

    test_ds = load_from_disk(config.TEST_DATA_LOAD_PATH)
    if config.num_test_samples and config.num_test_samples > 0:
        test_ds = test_ds.select(range(min(config.num_test_samples, len(test_ds))))
    
    test_ds = test_ds.map(format_to_messages).map(_sft_map_fn)
    test_ds = test_ds.select_columns(["input_ids", "labels", "assistant_mask"])

    scheduler = make_alpha_scheduler(config.scheduler)

    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

    args = MDLMConfig(
        push_to_hub=False,
        output_dir=config.TRAIN_MODEL_SAVE_PATH,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        time_epsilon=config.time_epsilon,
        loss_weight_type=config.loss_weight_type,
        dataloader_num_workers=config.num_workers,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,        
        save_strategy=config.save_strategy,
        report_to=config.report_to,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        bf16=True,                  

    )
    metric_computer = NLLPPLMetricComputer()

    trainer = MDLMSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=collator,
        scheduler=scheduler,
        compute_metrics=metric_computer,
    )

    trainer.train()
    #trainer.save_model()
    del trainer, args, collator, scheduler, train_ds, test_ds, model, tokenizer

    if torch.device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

