import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def setup_model_and_tokenizer(
    model_name: str,
    for_training: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = chat_template_str

    special_tokens_dict = {
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right" if for_training else "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device_map
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    return model, tokenizer
