import math
from pathlib import Path

import torch
from huggingface_hub import download_bucket_files
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.paths import PathResolver

DEFAULT_PATHS = PathResolver()


def resize_mdlm_vocab(model, new_vocab: int) -> None:
    backbone = model.backbone
    in_emb = backbone.vocab_embed.embedding  # nn.Parameter [V, H]
    out_lin = backbone.output_layer.linear  # nn.Linear(H, V)

    old_vocab, hidden = in_emb.shape
    assert out_lin.weight.shape == (old_vocab, hidden)
    assert out_lin.bias.shape == (old_vocab,)
    if new_vocab == old_vocab:
        return
    if new_vocab < old_vocab:
        raise ValueError(f"shrinking vocab not supported ({old_vocab} -> {new_vocab})")

    device_, dtype_ = in_emb.device, in_emb.dtype

    # --- input embedding ---------------------------------------------------
    new_in = torch.empty((new_vocab, hidden), device=device_, dtype=dtype_)
    torch.nn.init.kaiming_uniform_(new_in, a=math.sqrt(5))  # match EmbeddingLayer init
    with torch.no_grad():
        new_in[:old_vocab] = in_emb.data
    backbone.vocab_embed.embedding = torch.nn.Parameter(new_in)

    # --- output projection -------------------------------------------------
    new_w = torch.zeros((new_vocab, hidden), device=device_, dtype=out_lin.weight.dtype)
    new_b = torch.zeros((new_vocab,), device=device_, dtype=out_lin.bias.dtype)
    with torch.no_grad():
        new_w[:old_vocab] = out_lin.weight.data
        new_b[:old_vocab] = out_lin.bias.data
    out_lin.weight = torch.nn.Parameter(new_w)
    out_lin.bias = torch.nn.Parameter(new_b)
    out_lin.out_features = new_vocab

    model.config.vocab_size = new_vocab


def load_model(
    bucket: str = "avgJo3/mdlm-owt-bucket",
    local_dir: str | Path = DEFAULT_PATHS.resolve_weights_path("base"),
    tokenizer_name: str = "gpt2",
    verbose: bool = True,
):
    """
    Load the MDLM checkpoint + matching tokenizer, attach the chat template,
    add ChatML special tokens, then grow the model vocab to match.

    Order of operations (do NOT reorder):
      1. Download checkpoint artifacts from the bucket.
      2. Load the model from the local snapshot.
      3. Build tokenizer  ->  attach chat template  ->  add special tokens
         ->  set pad token. This finalizes len(tokenizer) BEFORE resize.
      4. Snapshot the MASK row at id 50257.
      5. Resize the model vocab to the next multiple of 64 >= len(tokenizer).
      6. Verify shapes and that the MASK row is byte-identical.
    """

    # --- templates & special tokens (encapsulated) -------------------------
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

    # extra_special_tokens = [
    #    "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|system|>",
    # ]

    # 1. ---- download checkpoint artifacts ---------------------------------
    resolver = DEFAULT_PATHS
    resolved_local_dir = resolver.resolve_weights_path(local_dir)
    resolved_local_dir.mkdir(parents=True, exist_ok=True)
    files = resolver.mdlm_checkpoint_files(resolved_local_dir)

    missing = [
        (filename, str(path)) for filename, path in files.items() if not path.exists()
    ]
    if missing:
        if verbose:
            print(f"Downloading {len(missing)} missing file(s)...")
        download_bucket_files(bucket, files=missing)
    elif verbose:
        print("All checkpoint files already present, skipping download.")

    # 2. ---- load model ----------------------------------------------------
    model = AutoModelForMaskedLM.from_pretrained(
        str(resolved_local_dir),  # or absolute path
        trust_remote_code=True,
    )
    print(next(model.parameters()).dtype)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS supports bfloat16 on M-series chips
        model = model.to(torch.bfloat16)
    else:
        print("bfloat16 not supported, keeping float32")
    # 3. ---- tokenizer: base -> template -> specials -> pad ----------------
    tokenizer_cache_dir = resolver.tokenizer_cache_dir(resolved_local_dir)
    if tokenizer_cache_dir.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_cache_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_cache_dir))

    if verbose:
        print(f"[tok] base tokenizer      : {tokenizer_name}")
        print(f"[tok] base vocab size     : {len(tokenizer)}")
        print(
            f"[tok] bos_token           : {tokenizer.bos_token!r}  (id {tokenizer.bos_token_id})"
        )
        print(
            f"[tok] eos_token           : {tokenizer.eos_token!r}  (id {tokenizer.eos_token_id})"
        )
        print(
            f"[tok] unk_token           : {tokenizer.unk_token!r}  (id {tokenizer.unk_token_id})"
        )
        print(
            f"[tok] pad_token (before)  : {tokenizer.pad_token!r}  (id {tokenizer.pad_token_id})"
        )
        print(
            f"[tok] mask_token (before) : {tokenizer.mask_token!r}  (id {tokenizer.mask_token_id})"
        )
        base_special = tokenizer.all_special_tokens
        print(f"[tok] special tokens ({len(base_special)}) : {base_special}")

    tokenizer.chat_template = chat_template_str
    tokenizer.add_special_tokens(
        {
            "mask_token": "<mask>",  # <-- gets id 50257 (first free slot)
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|user|>",
                "<|assistant|>",
                "<|system|>",
            ],
        }
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if verbose:
        print(f"[tok] vocab size after adding specials: {len(tokenizer)}")
        print(
            f"[tok] mask_token (after)  : {tokenizer.mask_token!r}  (id {tokenizer.mask_token_id})"
        )
        print(
            f"[tok] pad_token  (after)  : {tokenizer.pad_token!r}  (id {tokenizer.pad_token_id})"
        )
        all_special = tokenizer.all_special_tokens
        print(f"[tok] all special tokens ({len(all_special)}):")
        for tok in all_special:
            print(f"        {tok!r:25s}  id={tokenizer.convert_tokens_to_ids(tok)}")

    # 4. ---- verify pretrained vocab and snapshot the MASK row -------------
    old_vocab = model.backbone.vocab_embed.embedding.shape[0]
    if verbose:
        print(f"[mdl] pretrained vocab size: {old_vocab}")
    assert old_vocab >= 50258, "checkpoint smaller than expected"
    mask_row_before = model.backbone.vocab_embed.embedding[50257].detach().clone().cpu()

    # 5. ---- grow model vocab to next multiple of 64 >= len(tokenizer) -----
    padded_vocab = math.ceil(len(tokenizer) / 64) * 64
    resize_mdlm_vocab(model, padded_vocab)

    # 6. ---- post-conditions ----------------------------------------------
    new_vocab = model.backbone.vocab_embed.embedding.shape[0]
    if verbose:
        print(f"[mdl] tokenizer vocab size : {len(tokenizer)}")
        print(f"[mdl] resized vocab size   : {new_vocab}  (padded to multiple of 64)")
    assert new_vocab == padded_vocab
    assert new_vocab >= len(tokenizer)
    assert model.backbone.output_layer.linear.weight.shape == (
        new_vocab,
        model.config.hidden_dim,
    )

    mask_row_after = model.backbone.vocab_embed.embedding[50257].detach().clone().cpu()
    assert torch.equal(
        mask_row_before, mask_row_after
    ), "MASK embedding row changed during resize — append-only invariant broken"

    return model, tokenizer


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model, tokenizer = load_model(local_dir=tmp)
