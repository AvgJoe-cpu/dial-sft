from typing import Optional
import sys

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    class BlockMask:
        pass


class A2DGPTNeoXConfig(transformers.GPTNeoXConfig):  
    model_type = "a2d-gpt-neox"


class A2DGPTNeoXModel(transformers.GPTNeoXModel):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        """
        # -------------------------------------------------------------
        # ORIGINAL CODE (causal mask)
        # -------------------------------------------------------------
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        # -------------------------------------------------------------
        # ORIGINAL CODE (causal mask)
        # -------------------------------------------------------------
        """
        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------
        # 1) If no mask is provided → treat all tokens as valid (no padding)
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # 2) If mask is not already a 4D attention mask → convert it
        if not (
            isinstance(attention_mask, BlockMask)
            or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
        ):
            attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

        causal_mask = attention_mask
        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------

        hidden_states = self.emb_dropout(inputs_embeds)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                layer_past=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # installed GPTNeoXLayer returns (hidden_states,) tuple, not a bare tensor
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        hidden_states = self.final_layer_norm(hidden_states)


        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class A2DGPTNeoXForCausalLM(transformers.GPTNeoXForCausalLM):
    config: A2DGPTNeoXConfig

    def __init__(self, config):
        transformers.GPTNeoXPreTrainedModel.__init__(self, config)
        self.gpt_neox = A2DGPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


transformers.AutoConfig.register("a2d-gpt-neox", A2DGPTNeoXConfig)
transformers.AutoModel.register(A2DGPTNeoXConfig, A2DGPTNeoXForCausalLM)
transformers.AutoModelForMaskedLM.register(A2DGPTNeoXConfig, A2DGPTNeoXForCausalLM)


# ---------------------------------------------------------------------------
# Conversion utility  (previously convert.py)
# ---------------------------------------------------------------------------
def convert(model_name_or_path: str, output_dir: str, random_init: bool = False):
    """Convert a stock GPT-NeoX checkpoint to A2DGPTNeoX format."""
    src_model     = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    src_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    cfg_dict = src_model.config.to_dict()
    for k in ("model_type", "auto_map", "architectures"):
        cfg_dict.pop(k, None)
    tgt_config = A2DGPTNeoXConfig(**cfg_dict)
    tgt_model  = A2DGPTNeoXForCausalLM(tgt_config).to(torch.bfloat16)

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