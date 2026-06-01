import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model_and_tokenizer(
    model_name: str,
    for_training: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    local_dir: str = "./artifacts/ar_weights",
):
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

    # ── tokenizer: load from cache or download & save ─────────────────────
    tokenizer_cache_dir = os.path.join(local_dir, "tokenizer")
    if os.path.isdir(tokenizer_cache_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.chat_template = chat_template_str
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|user|>",
                    "<|assistant|>",
                    "<|system|>",
                ]
            }
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        os.makedirs(tokenizer_cache_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_cache_dir)

    tokenizer.padding_side = "right" if for_training else "left"

    # ── model ─────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device_map
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    return model, tokenizer
