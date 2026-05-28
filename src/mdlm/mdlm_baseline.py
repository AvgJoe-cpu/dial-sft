from src.mdlm.load_model import load_model
from src.mdlm.mdlm_helpers.mdlm_scheduler import BaseAlphaScheduler, LinearAlphaScheduler
from src.mdlm.mdlm_helpers.mdlm_trainer_sft import SFTCollator, MDLMConfig, MDLMSFTTrainer
from src.mdlm.mdlm_helpers.mdlm_sampler_sft import MinimalMDLMSampler, SFTMixinBatchedVarlen, MDLMSamplerConfig

import dataclasses
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#dd = load_dataset("euclaise/writingprompts")
#train_ds = dd['train'].select(range(10)).rename_column('story', 'completion')
#model, tokenizer = load_model(local_dir="./weights/base")


def format_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]}
        ]
    }
############

def run_training(
    TRAIN_DATA_LOAD_PATH: str,
    TRAIN_MODEL_LOAD_PATH: str,
    TRAIN_MODEL_SAVE_PATH: str = ".weights/checkpoints",
    num_samples: int = 100,
    num_epochs: int = 1,
    batch_size: int = 8,    
):
    

    def _sft_map_fn(example, max_length=512):
        enc = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            max_length=max_length,
            truncation=True,
        )
        input_ids      = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
        labels = [tok if m == 1 else -100 for tok, m in zip(input_ids, assistant_mask)]
        return {"input_ids": input_ids, "labels": labels, "assistant_mask": assistant_mask}    
    
    train_ds = load_from_disk(TRAIN_DATA_LOAD_PATH)

    print(f"Selecting {num_samples} samples and preprocessing...")
    train_ds = train_ds.select(range(num_samples)).rename_column('story', 'completion')    

    model, tokenizer = load_model(local_dir=TRAIN_MODEL_LOAD_PATH)
    train_ds = train_ds.map(format_to_messages)
    train_ds = train_ds.map(_sft_map_fn)
    train_ds = train_ds.select_columns(["input_ids", "labels", "assistant_mask"])

    scheduler = LinearAlphaScheduler()
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

    args = MDLMConfig(
        output_dir=TRAIN_MODEL_SAVE_PATH,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        batch_eval_metrics=True,
        remove_unused_columns=False,
    )

    trainer = MDLMSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=collator,        
        scheduler=scheduler
    )

    trainer.train()
    trainer.save_model()
    ##### del 
    ##### do gc 


def generate_mdlm(
    batch,
    tokenizer=None,
    model=None,
    sampler=None,                                # [VARLEN] reuse across batches
    config: MDLMSamplerConfig = MDLMSamplerConfig(),
):
    messages_list = [[{"role": "user", "content": p}] for p in batch["prompt"]]

    encoded = tokenizer.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    prompt_ids = encoded["input_ids"].to(model.device)        # [B, P_max]
    attn       = encoded["attention_mask"].to(model.device)   # [B, P_max], 1=real
    prompt_lens = attn.sum(dim=1).long()                       # [B]  [VARLEN]

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, (
        "tokenizer has no pad_token_id; set tokenizer.pad_token = tokenizer.eos_token "
        "or similar before generation."
    )
    assert pad_id != tokenizer.mask_token_id, (
        "tokenizer.pad_token_id == mask_token_id; SUBS would treat pad as response."
    )

    out = sampler.sample_sft(
        prompt_ids,
        prompt_lens=prompt_lens,                              # [VARLEN]
        pad_token_id=pad_id,                                  # [VARLEN]
        **dataclasses.asdict(config),
    )                                                          # [B, P_max + R]

    # [VARLEN] per-row response slice: response_b = out[b, P_b : P_b + R]
    R = config.response_length
    decoded = [
        tokenizer.decode(
            out[b, int(prompt_lens[b]) : int(prompt_lens[b]) + R],
            skip_special_tokens=True,
        )
        for b in range(out.shape[0])
    ]
    return {"gen": decoded}


def run_inference_mdlm():

    #dataset = load_from_disk(INFERENCE_LOAD_PATH)

    #del dataset
    #ds = ds.select(range(num_samples))    

    model, tokenizer = load_model()
    config = MDLMSamplerConfig(response_length=20, num_steps=20)

    sampler = MinimalMDLMSampler(
        backbone=model.eval(),
        scheduler=LinearAlphaScheduler(),
        mask_index=tokenizer.mask_token_id,
    )
    sampler.sample_sft = SFTMixinBatchedVarlen.sample_sft.__get__(sampler, type(sampler))

    ds = ds.map(
        generate_mdlm,
        batched=True,
        batch_size=2,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model":     model,
            "sampler":   sampler,
            "config":    config,
        },
    )
    return ds 

    ##ds.save_to_disk(SAVE_PATH)
    ##torch.cuda.empty_cache()
    ##del model, tokenizer, config, ds, INFERENCE_LOAD_PATH, SAVE_PATH, MODEL_NAME


if __name__ == "__main__":
    TRAIN_OG_PATH       = "euclaise/writingprompts"
    TRAIN_OG_SAVE_PATH  = "./datasets/base/train"
    EVAL_OG_SAVE_PATH   = "./datasets/base/eval"
    TEST_OG_SAVE_PATH   = "./datasets/base/test"

    from src.ar.ar_baseline import setup_model_and_tokenizer, process_and_save_datasets, count_tokens_in_column_batched

    _, tokenizer = setup_model_and_tokenizer(model_name="EleutherAI/pythia-70m")
    process_and_save_datasets(
        dataset_name=TRAIN_OG_PATH,
        tokenizer=tokenizer,
        train_save_path=TRAIN_OG_SAVE_PATH,
        eval_save_path=EVAL_OG_SAVE_PATH,
        test_save_path=TEST_OG_SAVE_PATH,
    )
    
    del tokenizer

    CONFIG_DICT = {
        "ROUND1": {
            "MDLM_TRAIN_DATA_LOAD_PATH":  TRAIN_OG_SAVE_PATH,    # "./datasets/base/train"
            "MDLM_TRAIN_MODEL_LOAD_PATH": "./weights/base",
            "MDLM_TRAIN_MODEL_SAVE_PATH": "./weights/checkpoints",
        },
    }
    for round_name, config in CONFIG_DICT.items():
            
        train_data_load_path  = config["MDLM_TRAIN_DATA_LOAD_PATH"]
        train_model_load_path = config["MDLM_TRAIN_MODEL_LOAD_PATH"]
        train_model_save_path = config["MDLM_TRAIN_MODEL_SAVE_PATH"]
    
        run_training(
            TRAIN_DATA_LOAD_PATH=train_data_load_path,
            TRAIN_MODEL_LOAD_PATH=train_model_load_path,
            TRAIN_MODEL_SAVE_PATH=train_model_save_path,
            num_samples=100,
        )
        print(f"✓ [{round_name}] Training complete\n")
