from src.ar.ar_load_model import setup_model_and_tokenizer
from src.ar.ar_config_schema import InferenceConfig

import torch
from transformers import GenerationConfig
from datasets import load_from_disk


def generate_ar(batch, tokenizer=None, model=None, gen_config=None):
    messages_list = [
        [{"role": "user", "content": prompt}] for prompt in batch["prompt"]
    ]
    formatted_texts = tokenizer.apply_chat_template(
        messages_list, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        generation_config=gen_config,
    )

    results = tokenizer.batch_decode(
        [
            generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
            for i in range(len(generated_ids))
        ],
        skip_special_tokens=True,
    )

    del model_inputs, generated_ids, formatted_texts, messages_list
    return {"story": results}


def run_inference(cfg: InferenceConfig) -> None:
    ds = load_from_disk(cfg.INFER_DATA_LOAD_PATH)
    ds = ds.select(range(cfg.num_samples))

    model, tokenizer = setup_model_and_tokenizer(
        model_name=cfg.INFER_MODEL_LOAD_PATH, for_training=False
    )
    tokenizer.padding_side = "left"

    config = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        num_beams=cfg.num_beams,
        do_sample=cfg.do_sample,
        use_cache=cfg.use_cache,
        temperature=cfg.temperature,
        num_return_sequences=cfg.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    ds = ds.map(
        generate_ar,
        batched=True,
        batch_size=cfg.batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "gen_config": config,
        },
    )
    ds.save_to_disk(cfg.INFER_DATA_SAVE_PATH)
    torch.cuda.empty_cache()
    del model, tokenizer, config, ds
