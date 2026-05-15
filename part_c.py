import torch
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import re
from jinja2 import Template
from datasets import load_dataset

def pre_split(batch, in_key="utterances"):
    utts = batch[in_key]
    return {"first_pair": [utt[:2] for utt in utts],"second_pair": [utt[2:] for utt in utts]}

def ar_format(example, template=None):
    first_pairs = example["first_pair"]
    second_pairs = example["second_pair"]
    texts = []
    for fp, sp in zip(first_pairs, second_pairs):
        rendered = template.render(
            first_pair=fp,
            second_pair=sp
        ).strip()
        texts.append(rendered)
    return {"text": texts}


def ar_render(example, tokenizer=None):
    example["text"] = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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


def parse_generated_response(example):
    response = example.get("response", "")
    if not response:
        return None

    # Regex to match content between ```json and ```
    match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if not match:
        return None

    json_str = match.group(1).strip()
    if not json_str:
        print("Empty JSON content in response")
        return None

    try:
        data = json.loads(json_str)
        return {"parsed": data}
    except json.JSONDecodeError:
        print("Invalid JSON content in response")
        return None


def parse_analysis_content(example):
    parsed_data = example.get("parsed")
    if parsed_data is None:
        return None

    description = parsed_data.get("description", "")
    explanation = parsed_data.get("explanation", [])
    final_answer = parsed_data.get("final_answer", "")

    explanation_steps = []
    if isinstance(explanation, list):
        explanation_steps = [step.strip() for step in explanation if isinstance(step, str)]
    elif isinstance(explanation, str):
        explanation_steps = [line.strip("- ").strip() for line in explanation.split("\n") if line.strip()]

    return {
        "description": description,
        "explanation_steps": explanation_steps,
        "final_answer": final_answer
    }



def run_syn(
            checkpoint_path="Qwen/Qwen2.5-3B-Instruct",
            dataset_name="roskoN/dailydialog",
            split="validation",
            inference_batch_size=128,
            output_file="synthetic_analysis.parquet",            
):
    set_seed(42)
    # BASELINE
    pilot_template = Template("""
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

    """)

    dd = load_dataset(dataset_name)
    dataset = dd[split]

    instruct_path = checkpoint_path
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_path,
        device_map="auto",
        dtype=torch.float16,  #
    )
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_path)
    instruct_config = GenerationConfig(
        max_new_tokens = 4096,
        max_length     = 128,
        num_beams      = 1,     # multinomial
        do_sample      = True,  #
        use_cache      = True,
        temperature    = 1.1,

        num_return_sequences = 1,
        pad_token_id         = instruct_tokenizer.pad_token_id,
        eos_token_id         = instruct_tokenizer.eos_token_id,
        bos_token_id         = instruct_tokenizer.bos_token_id,

    )
    dataset = dataset.map(
        pre_split,
        batched=True,
        fn_kwargs={
            "in_key": "utterances"
        }
    ).map(
        ar_format,
        batched=True,
        fn_kwargs={
            "template": pilot_template
        }
    ).map(
        ar_render,
        batched=True,
        fn_kwargs={
            "tokenizer": instruct_tokenizer
        }
    ).map(
        generate_ar,
        batched=True,
        batch_size=inference_batch_size,
        fn_kwargs={
            "tokenizer": instruct_tokenizer,
            "model": instruct_model,
            "gen_config": instruct_config,
            "in_key": "text",
            "out_key": "response"
        }
    )    

    dataset = dataset.map(
        parse_generated_response,

    )
    dataset = dataset.map(
        parse_analysis_content,

    )    

    dataset = dataset.remove_columns(["emotions", "acts", "text", "response", "parsed"])    
    dataset.to_parquet(output_file)

    del instruct_model, instruct_tokenizer, instruct_config, dataset, dd
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_syn()    
