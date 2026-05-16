from src.data_posprocess import parse_content, parse_generated_response
from..data_preprocess import pre_process

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


def run_inference(
        checkpoint_path="checkpoint-500",
        dataset_name="roskoN/dailydialog",
        split="train",
        inference_batch_size=128,
        output_file="inference_output.parquet",
        two_inputs=True,
        config: GenerationConfig | None = None,
        post_process: bool = False
):

    dd = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if two_inputs:
        temp_user = """
        Task: Analyze the two conversation snippets provided below—an initial snippet and its continuation.
        Given a conversation snippet:
        {%- for utt in initial_utterances %}
        {{ utt }}
        {%- if not loop.last %}\n{%- endif %}
        {%- endfor %}
        And its continuation:
        {%- for utt in continuation_utterances %}
        {{ utt }}
        {%- if not loop.last %}\n{%- endif %}
        {%- endfor %}
        First, briefly describe the conversation snippet and its continuation.
        Then, explain why the conversation continues as it does, based only on the provided snippet and continuation.
        Answer briefly, in steps.
        Keep the explanation minimal and focus each step on one key reason.
        Output your analysis as a valid JSON object with exactly the following structure:
        ```json
        {
        "description": "Brief description of the snippet and continuation",
        "explanation": "Brief explanation in steps why the conversation continues",
        "final_answer": "Concise summary of key reasons"
        }
        ```
        """        
        ds = pre_process(
            dd[split],
            tokenizer=tokenizer,
            user_template=temp_user,
            inference_type="joint",
            mode="inference",
        )
    else: 
         # --- Templates ---    
        temp_user = """
        Continue the following dialogue:
        {%- for utt in utterances %}
        {{ utt }}
        {%- if not loop.last %}\n{%- endif %}
        {%- endfor %}
        """        

        ds = pre_process(
            dd[split],
            tokenizer=tokenizer,
            user_template=temp_user,
            inference_type="separate",
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
    if post_process:
        ds = ds.map(
            parse_generated_response,
            batched=False,
        )
        ds = ds.map(
            parse_content,
            batched=False,
        )

    ds.to_parquet(output_file)
    del model, tokenizer, config, ds, dd
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_inference()    