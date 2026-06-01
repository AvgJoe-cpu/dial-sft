import dataclasses
import gc
from dataclasses import dataclass

import torch
from datasets import load_from_disk

from src.mdlm.mdlm_helpers.mdlm_sampler_sft import (
    MDLMSamplerConfig,
    MinimalMDLMSampler,
    SFTMixinBatchedVarlen,
)
from src.mdlm.mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler
from src.mdlm.mdlm_load_model import load_model
from src.paths import PathResolver

DEFAULT_PATHS = PathResolver()


@dataclass
class InferenceConfig:
    # --- data & model paths ---
    INFERENCE_LOAD_PATH: str = str(
        DEFAULT_PATHS.resolve_dataset_path("writingprompts_test")
    )
    INFERENCE_SAVE_PATH: str = str(
        DEFAULT_PATHS.resolve_weights_path("mdlm/inference_outputs")
    )
    INFERENCE_MODEL_PATH: str = str(DEFAULT_PATHS.resolve_weights_path("base"))

    # --- dataset ---
    num_samples: int = 10
    batch_size: int = 2

    # --- MDLMSamplerConfig contract ---
    response_length: int = 20
    num_steps: int = 20


def generate_mdlm(
    batch,
    tokenizer=None,
    model=None,
    sampler=None,
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
    prompt_ids = encoded["input_ids"].to(model.device)  # [B, P_max]
    attn = encoded["attention_mask"].to(model.device)  # [B, P_max], 1=real
    prompt_lens = attn.sum(dim=1).long()  # [B]  [VARLEN]

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, (
        "tokenizer has no pad_token_id; set tokenizer.pad_token = tokenizer.eos_token "
        "or similar before generation."
    )
    assert (
        pad_id != tokenizer.mask_token_id
    ), "tokenizer.pad_token_id == mask_token_id; SUBS would treat pad as response."

    out = sampler.sample_sft(
        prompt_ids,
        prompt_lens=prompt_lens,  # [VARLEN]
        pad_token_id=pad_id,  # [VARLEN]
        **dataclasses.asdict(config),
    )  # [B, P_max + R]

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


def run_inference_mdlm(config: InferenceConfig = InferenceConfig()):

    model, tokenizer = load_model(local_dir=config.INFERENCE_MODEL_PATH)
    sampler_config = MDLMSamplerConfig(
        response_length=config.response_length,
        num_steps=config.num_steps,
    )

    sampler = MinimalMDLMSampler(
        backbone=model.eval(),
        scheduler=LinearAlphaScheduler(),
        mask_index=tokenizer.mask_token_id,
    )
    sampler.sample_sft = SFTMixinBatchedVarlen.sample_sft.__get__(
        sampler, type(sampler)
    )

    ds = load_from_disk(config.INFERENCE_LOAD_PATH)
    ds = ds.select(range(config.num_samples))

    ds = ds.map(
        generate_mdlm,
        batched=True,
        batch_size=config.batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "sampler": sampler,
            "config": sampler_config,
        },
    )

    ds.save_to_disk(config.INFERENCE_SAVE_PATH)

    del model, tokenizer, sampler, sampler_config, ds
    gc.collect()
    if torch.device.type == "cuda":
        torch.cuda.empty_cache()
