from src.ar.ar_load_model import setup_model_and_tokenizer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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
            generated_ids[i][len(model_inputs.input_ids[i]) :].tolist()
            for i in range(len(generated_ids))
        ],
        skip_special_tokens=True,
    )

    del model_inputs, generated_ids, formatted_texts, messages_list
    return {"story": results}


def run_inference(
    INFER_MODEL_LOAD_PATH,
    INFER_DATA_LOAD_PATH,
    INFER_DATA_SAVE_PATH,
    num_samples: int = 10000,
):
    MODEL_NAME = INFER_MODEL_LOAD_PATH
    INFERENCE_LOAD_PATH = INFER_DATA_LOAD_PATH
    SAVE_PATH = INFER_DATA_SAVE_PATH

    dataset = load_from_disk(INFERENCE_LOAD_PATH)
    ds = dataset
    del dataset
    ds = ds.select(range(num_samples))

    model, tokenizer = setup_model_and_tokenizer(
        model_name=MODEL_NAME, for_training=False
    )  # NEW
    tokenizer.padding_side = "left"

    config = GenerationConfig(
        max_new_tokens=512,
        num_beams=1,
        do_sample=True,
        use_cache=True,
        temperature=1.1,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )



    ds = ds.map(
        generate_ar,
        batched=True,
        batch_size=10,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "gen_config": config,
        },
        #    remove_columns=["text"]
    )
    ds.save_to_disk(SAVE_PATH)
    torch.cuda.empty_cache()
    del model, tokenizer, config, ds, INFERENCE_LOAD_PATH, SAVE_PATH, MODEL_NAME