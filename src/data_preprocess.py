from typing import Any, Literal
from jinja2 import Template

def pre_split(example, in_key="utterances"):
    first_pair = example[in_key][:1]
    second_pair = example[in_key][1:]
    return {"first_pair": first_pair, "second_pair": second_pair}


def pre_format(
    example,
    user_template: str | None = None,
    assistant_template: str | None = None,
    joint_template: str | None = None,
    tokenizer: Any | None = None,
    mode: Literal["training", "inference"] = "training",
    inference_type: Literal["separate", "joint"] = "separate",
    out_key: str = "text",
    in_keys: str | tuple[str, str] = ("first_pair", "second_pair"),
):
    if mode not in ("training", "inference"):
        raise ValueError(f"Invalid mode: {mode!r}. Expected 'training' or 'inference'.")
    
    if inference_type not in ("separate", "joint"):
        raise ValueError(f"Invalid inference_type: {inference_type!r}. Expected 'separate' or 'joint'.")
    
    if mode == "training" and inference_type == "joint":
        raise ValueError("inference_type='joint' is only valid when mode='inference'.")
  
    if isinstance(in_keys, str):
        if mode == "training" or inference_type == "joint":
            raise ValueError(f"str in_keys requires mode='inference' + inference_type='separate', got ({mode!r}, {inference_type!r}).")
        first_pair, second_pair = example[in_keys], None
        
    elif isinstance(in_keys, tuple):
        if inference_type == "separate" and mode == "inference":
            raise ValueError("tuple in_keys requires mode='training' or inference_type='joint'.")
        first_pair, second_pair = example[in_keys[0]], example[in_keys[1]]
    else:
        raise TypeError(f"in_keys must be str or tuple[str, str], got {type(in_keys).__name__!r}.")
        
    if mode == "inference":
        if inference_type == "joint":
            # CASE: INFERENCE WITH TEMPLATE for 2 inputs
            joint_template = Template(joint_template)
            joint = joint_template.render(utterances1=first_pair, utterances2=second_pair).strip()  # Assuming utterances1 and utterances2 for first and second
            tokenized = tokenizer.apply_chat_template([{"role": "user", "content": joint}], tokenize=False, add_generation_prompt=True)
            return {out_key: tokenized}
        
        else:  # inference_type == "separate" or default
            # NESTED CASE: INFERENCE WITH TEMPLATE for 1 input  
            user_template = Template(user_template)
            user = user_template.render(utterances=first_pair).strip()
            tokenized = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            return {out_key: tokenized}
        
    else:  # mode == "training"
        # CASE: TRAINING WITH SEPARATE TEMPLATE
        user_template = Template(user_template)
        assistant_template = Template(assistant_template)  
        user = user_template.render(utterances=first_pair).strip()
        response = assistant_template.render(utterances=second_pair).strip()

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": response}
        ]        
        tokenized = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        return {out_key: tokenized}


def pre_process(
        ds,
        user_template: str | None = None,
        assistant_template: str | None = None,
        joint_template: str | None = None,
        tokenizer: Any | None = None,
        mode: str = "training",
        inference_type: str = "separate",
        out_key: str = "text",
        in_keys: str | tuple[str, str] = ("first_pair", "second_pair")
):        
    if mode == "training" and not (user_template and assistant_template):
        raise ValueError("Both user_template and assistant_template are required for mode='training'.")
    
    elif mode == "inference" and inference_type == "joint" and not joint_template:
        raise ValueError("joint_template is required for mode='inference' and inference_type='joint'.")
    
    elif mode == "inference" and inference_type == "separate" and not user_template:
        raise ValueError("user_template is required for mode='inference' and inference_type='separate'.")
    
    if not tokenizer:
        raise ValueError("tokenizer is required.")
        
    return (
        ds
        .map(pre_split, batched=False)
        .map(
            pre_format,
            batched=False,
            fn_kwargs={
                "user_template": user_template,
                "assistant_template": assistant_template,
                "joint_template": joint_template,
                "tokenizer": tokenizer,
                "mode": mode,
                "inference_type":inference_type,
                "out_key": out_key,
                "in_keys": in_keys,
            }, remove_columns=["utterances", "first_pair", "second_pair"]
        )            

    )

if __name__ == "__main__":
    print("This module is not meant to be run directly. Import it instead.")