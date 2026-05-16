from src.model.a2dgptox import A2DGPTNeoXConfig, A2DGPTNeoXForCausalLM

import sys
import torch
import transformers


def convert(model_name_or_path: str, output_dir: str, random_init: bool = False):
    src_model     = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    src_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    cfg_dict = src_model.config.to_dict()
    for k in ("model_type", "auto_map", "architectures"):
        cfg_dict.pop(k, None)
    tgt_config = A2DGPTNeoXConfig(**cfg_dict)
    tgt_model = A2DGPTNeoXForCausalLM(tgt_config).to(torch.bfloat16)

    if not random_init:
        missing, unexpected = tgt_model.load_state_dict(src_model.state_dict(), strict=False)
        print("missing:   ", missing)
        print("unexpected:", unexpected)

    tgt_model.save_pretrained(output_dir)
    tgt_config.save_pretrained(output_dir)
    src_tokenizer.save_pretrained(output_dir)
    print(f"saved to {output_dir}")

if __name__ == "__main__":
    convert(
        model_name_or_path=sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-70m",
        output_dir=sys.argv[2]         if len(sys.argv) > 2 else "models-tmp/a2d-gpt-neox-70m",
    )