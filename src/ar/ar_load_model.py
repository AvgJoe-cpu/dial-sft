from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.paths import PathResolver

DEFAULT_PATHS = PathResolver()


def setup_model_and_tokenizer(
    model_name: str,
    for_training: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    local_dir: str | Path = DEFAULT_PATHS.resolve_weights_path("ar"),
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

    resolver = DEFAULT_PATHS
    resolved_model_name = resolver.resolve_model_reference(model_name)

    # ── tokenizer: load from cache or download & save ─────────────────────
    tokenizer_cache_dir = resolver.tokenizer_cache_dir(local_dir)
    if tokenizer_cache_dir.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_cache_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
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
        tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_cache_dir))

    tokenizer.padding_side = "right" if for_training else "left"

    # ── model ─────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_name, dtype=dtype, device_map=device_map
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    return model, tokenizer
