# SFT.py (retrived from: https://huggingface.co/learn/llm-course/chapter11/3)
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from jinja2 import Template

def pre_split(batch, in_key="utterances"):
    utts = batch[in_key]
    return {"first_pair": [utt[:2] for utt in utts],"second_pair": [utt[2:] for utt in utts]}


def pre_format(example, user_template=None, assistant_template=None):
    return {
        "user": [user_template.render(utterances=fp).strip() for fp in example["first_pair"]],           # Comprehension for user renders
        "response": [assistant_template.render(utterances=sp).strip() for sp in example["second_pair"]]  # Comprehension for assistant renders
    }

def pre_template(example):
    return {
        "text": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": u},
                {"role": "assistant", "content": r}
            ]
            for u, r in zip(example["user"], example["response"])  # Comprehension over the batch
        ]
    }


def pre_render(batch, tokenizer=None):
    messages_list = batch["text"]  # List of lists: each item is a list of message dicts

    formatted_texts = []
    for messages in messages_list:
        formatted_text = tokenizer.apply_chat_template(
            messages,  # List of dicts for this example
            tokenize=False,  # Return string, not tokens
            add_generation_prompt=False  # For SFT, no extra prompt
        )
        formatted_texts.append(formatted_text)

    return {"text": formatted_texts}  # Return dict with updated column



def process_dataset(ds, temp_user, temp_assistant, tokenizer):
    return (
        ds
        .map(pre_split, batched=True, fn_kwargs={"in_key": "utterances"})
        .map(pre_format, batched=True, fn_kwargs={
            "user_template": temp_user,
            "assistant_template": temp_assistant
        })
        .map(pre_template, batched=True)
        .map(pre_render, batched=True, fn_kwargs={
            "tokenizer": tokenizer
        }, remove_columns=["utterances", "first_pair", "second_pair", "user", "response", "acts", "emotions"])
    )



def run_training(
    model_name: str = "facebook/opt-350m",
    dataset_name: str = "roskoN/dailydialog",
    split: str = "train",
    output_dir: str = "./sft_output",
):
    # --- Templates ---
    temp_user = Template("""
    Continue the following dialogue:
    {%- for utt in utterances %}
    {{ utt }}
    {%- if not loop.last %}\n{%- endif %}
    {%- endfor %}
    """)
    temp_assistant = Template("""
    {%- for utt in utterances %}
    {{ utt }}
    {%- if not loop.last %}\n{%- endif %}
    {%- endfor %}
    """)
    chat_template_str = """
    {%- for message in messages %}
    {%- if message['role'] == 'system' %}
    {{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' %}
    {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
    {{- '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{- '<|im_end|>' }}
    {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- endif %}
    """.strip()
    special_tokens_dict = {
        "additional_special_tokens": ("<|im_start|>", "<|im_end|>")
    }

    # --- Tokenizer (before model: needed for preprocessing) ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.chat_template = chat_template_str

    # --- Preprocessing (model not yet in memory) ---
    dd = load_dataset(dataset_name)
    train_ds = process_dataset(
        ds=dd[split],
        temp_user=temp_user,
        temp_assistant=temp_assistant,
        tokenizer=tokenizer,
    )
    del dd
    train_ds = train_ds.select(range(1000))  

    # --- Model (loaded after preprocessing to avoid idle VRAM) ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # --- Training ---
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=500,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        auto_find_batch_size=True,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        bf16=True,
        completion_only_loss=False,
        optim="adamw_torch_fused",
        learning_rate=1e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    # --- Teardown ---
    del trainer, model
    torch.cuda.empty_cache()