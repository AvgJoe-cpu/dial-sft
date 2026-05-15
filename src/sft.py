from src.preprocess import pre_process as preprocess_function

# SFT.py (retrived from: https://huggingface.co/learn/llm-course/chapter11/3)
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

def run_training(
    model_name: str = "facebook/opt-350m",
    dataset_name: str = "roskoN/dailydialog",
    split: str = "train",
    output_dir: str = "./sft_output",
    training_args: SFTConfig | None = None,  # New optional parameter

):
    # --- Templates ---
    temp_user = """
    Continue the following dialogue:
    {%- for utt in utterances %}
    {{ utt }}
    {%- if not loop.last %}\n{%- endif %}
    {%- endfor %}
    """
    temp_assistant = """
    {%- for utt in utterances %}
    {{ utt }}
    {%- if not loop.last %}\n{%- endif %}
    {%- endfor %}
    """
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
    train_ds = preprocess_function(
        dd[split],
        tokenizer=tokenizer,
        user_template=temp_user,
        assistant_template=temp_assistant,
        mode="training",        
        inference_type="separate"   # ← add this
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
    if training_args is None:  # Use defaults if not provided
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
    
if __name__ == "__main__":
    run_training()    