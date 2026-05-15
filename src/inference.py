from.preprocess import pre_process

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset

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
        checkpoint_path="checkpoint-500",
        dataset_name="roskoN/dailydialog",
        split="train",
        inference_batch_size=128,
        output_file="inference_output.parquet",
        two_inputs=True,
        config: GenerationConfig | None = None
):
    # --- Templates ---    
    temp_user = """
    Continue the following dialogue:
    {%- for utt in utterances %}
    {{ utt }}
    {%- if not loop.last %}\n{%- endif %}
    {%- endfor %}
    """

    dd = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if two_inputs:
        ds = pre_process(
            dd[split],
            tokenizer=tokenizer,
            user_template=temp_user,
            inference_type="separate",
            mode="inference",
        )
    else: 
        ds = pre_process(
            dd[split],
            tokenizer=tokenizer,
            user_template=temp_user,
            inference_type="joint",
            mode="inference",
        )

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    if config is None:
        config = GenerationConfig(
            max_new_tokens = 512,
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
    ds = ds.map(
        generate_ar,
        batched=True,            
        batch_size=inference_batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "gen_config": config,
            "in_key": "text",
            "out_key": "generated_response"
        }
    )    

    ds.to_parquet(output_file)
    del model, tokenizer, config, ds, dd
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_inferene()    