import torch
from transformers import  set_seed, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from jinja2 import Template

def pre_split(batch, in_key="utterances"):
    utts = batch[in_key]
    return {"first_pair": [utt[:2] for utt in utts],"second_pair": [utt[2:] for utt in utts]}


def pre_format_user(example, user_template=None):
    return {
        "text": [user_template.render(utterances=fp).strip() for fp in example["first_pair"]],          
    }


def ar_render_user(example, tokenizer=None):
    example["text"] = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in example["text"]
    ]
    return example


def generate_ar(example, tokenizer=None, model=None, gen_config=None, in_key=None, out_key=None):
    input_texts = example[in_key]
    generation_config = gen_config
    model_inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        generation_config=generation_config,
    )
    output_ids = [
        generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
        for i in range(len(generated_ids))
    ]
    content = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    example[out_key] = content
    del model_inputs, generated_ids, output_ids
    return example


def run_inferene(
        checkpoint_path="./sft_output/checkpoint-500",
        dataset_name="roskoN/dailydialog",
        split="train",
        inference_batch_size=128,
        output_file="inference_output.parquet"
):
    # --- Templates ---    
    temp_user = Template("""
    Continue the following dialogue:
    {%- for utt in utterances %}
    {{ utt }}
    {%- if not loop.last %}\n{%- endif %}
    {%- endfor %}
    """)


    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,   # Match training dtype (bf16=True)
        device_map="auto",            # Auto-place on GPU if available
    )
    model.eval()    

    gen_config = GenerationConfig(
        max_length     = 128,
        num_beams      = 1,     # multinomial
        do_sample      = True,  #
        use_cache      = True,
        temperature    = 1.1,

        num_return_sequences = 1,
        pad_token_id         = tokenizer.pad_token_id,
        eos_token_id         = tokenizer.eos_token_id,
        bos_token_id         = tokenizer.bos_token_id,

    )
    dd = load_dataset(dataset_name)
    ds = dd[split]

    ds = ds.map(
        pre_split,
        batched=True
    ).map(
        pre_format_user,
        fn_kwargs={"user_template": temp_user},
        batched=True
    ).map(
        ar_render_user,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True
    ).map(
        generate_ar,
        batched=True,
        batch_size=inference_batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "gen_config": gen_config,
            "in_key": "text",
            "out_key": "response",
        }
    )    
    ds.to_parquet(output_file)

    del model, tokenizer, gen_config, ds, dd
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    run_inferene()