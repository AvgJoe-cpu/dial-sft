from .configuration_mdlm import MDLMConfig

import math 

import torch 
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


## ROPE 
class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # persistent=True for checkpoint compatibility (modern convention is False,
        # but the published MDLM state-dict includes this buffer).
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)               # (T, Dh/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (T, Dh)
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
        self._seq_len_cached = seq_len

    def forward(self, seq_len: int, device, dtype):
        if (self._cos_cached is None
            or seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype  != dtype):
            self._build_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, H, T, Dh); cos, sin: (T, Dh) → broadcast to (1, 1, T, Dh)
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


## EMBED 
class TimestepEmbedder(nn.Module):
  def __init__(self, cond_dim: int, freq_dim: int = 256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(freq_dim, cond_dim, bias=True),
      nn.SiLU(),
      nn.Linear(cond_dim, cond_dim, bias=True))
    self.freq_dim = freq_dim

  def _fourier_features(self, t, max_period: int = 10_000):
    half = self.freq_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if self.freq_dim % 2:
      emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

  def forward(self, t):
    return self.mlp(self._fourier_features(t))
  

# LEGACY EMBEDDING
class EmbeddingLayer(nn.Module):
  def __init__(self, hidden_dim, vocab_size):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_size, hidden_dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]
  

## LM HEAD
class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_dim: int, vocab_size: int, cond_dim: int):
    super().__init__()
    self.norm_final   = nn.LayerNorm(hidden_dim, bias=False)
    self.linear       = nn.Linear(hidden_dim, vocab_size)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    return self.linear(modulate(self.norm_final(x), shift, scale))  
  

## TF BLOCK 
def modulate(x, shift, scale): return x * (1 + scale) + shift

class DDiTBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, cond_dim, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout  = dropout
        self.mlp_ratio = mlp_ratio        

        self.norm1 = nn.LayerNorm(hidden_dim, bias=False)   # PyTorch ≥ 2.1 supports `bias=False`
        self.norm2 = nn.LayerNorm(hidden_dim, bias=False)   

        self.mlp = nn.Sequential(
          nn.Linear(hidden_dim, mlp_ratio * hidden_dim, bias=True),
          nn.GELU(approximate='tanh'),
          nn.Linear(mlp_ratio * hidden_dim, hidden_dim, bias=True))        
        
        self.attn_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False) 
        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)     ### ATT OUT

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * hidden_dim, bias=True)         
        self.adaLN_modulation.weight.data.zero_()           
        self.adaLN_modulation.bias.data.zero_()             


    def forward(self, x, c, rotary_cos_sin, attention_mask=None):
        B, T, D = x.shape[0], x.shape[1], x.shape[2]

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:,None].chunk(6, dim=2)       

        x_skip  = x        
        x       = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv     = self.attn_qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        cos, sin = rotary_cos_sin
        q, k     = apply_rope(q, k, cos, sin)        #  new per-tensor RoPE
        att = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=attention_mask).transpose(1, 2).contiguous().reshape(B, T, D)

        # ---- attention sub-block ----
        x = x_skip + gate_msa * F.dropout(
            self.attn_out(att), p=self.dropout, training=self.training)

        # ---- MLP sub-block ----
        x = x + gate_mlp * F.dropout(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)),
            p=self.dropout, training=self.training)
        return x


## LM 
class DITBackbone(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.vocab_embed = EmbeddingLayer(config.hidden_dim, config.vocab_size)
    self.sigma_map   = TimestepEmbedder(config.cond_dim)
    self.rotary_emb  = Rotary(config.hidden_dim // config.n_heads)    

    self.blocks = nn.ModuleList([
    DDiTBlock(config.hidden_dim,
              config.n_heads,
              config.cond_dim,
              dropout=config.dropout)
    for _ in range(config.n_blocks)
    ])
    self.output_layer = DDitFinalLayer(config.hidden_dim, config.vocab_size, config.cond_dim)

  def forward(self, input_ids, sigma, attention_mask=None, output_hidden_states=False):  

    if not self.config.time_conditioning:
      sigma = torch.zeros_like(sigma)    

    all_hidden_states = []
    x = self.vocab_embed(input_ids)
    if output_hidden_states: all_hidden_states.append(x)

    c = F.silu(self.sigma_map(sigma))     
    rotary_cos_sin = self.rotary_emb(x.shape[1], x.device, x.dtype)
    # --- prepare attention mask once (bidirectional, padding-only) ---------
    # SDPA expects either None, a bool/float (B,*,T,T) bias, or to be told
    # is_causal=True. A (B,T) padding mask must be expanded to an additive
    # (B,1,1,T) bias with -inf on pad keys.
    if attention_mask is not None and attention_mask.dim() == 2:
        attention_mask = _prepare_4d_attention_mask(
            attention_mask, dtype=x.dtype
        )

    for i in range(len(self.blocks)):
        x = self.blocks[i](x, c ,rotary_cos_sin, attention_mask=attention_mask)
        if output_hidden_states: all_hidden_states.append(x)

    logits = self.output_layer(x, c)
    return logits, all_hidden_states     



class MDLM(transformers.PreTrainedModel):
    config_class = MDLMConfig
    base_model_prefix = "mdlm"
    _tied_weights_keys = []  # Explicitly declare no tied weights

    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.backbone = DITBackbone(config)
        # post_init() is called automatically by PreTrainedModel.from_pretrained()
        self.post_init()
        
    def forward(self, input_ids=None, timesteps=None, attention_mask=None, output_hidden_states=None, return_dict=None):
        # Use config defaults only if not provided
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Default timesteps if not provided
        if timesteps is None:
            timesteps = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.float32)

        # Forward pass
        logits, all_hidden_states = self.backbone(
            input_ids=input_ids,
            sigma=timesteps,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )

        # Return based on return_dict flag
        if return_dict:
            return transformers.modeling_outputs.MaskedLMOutput(
                logits=logits,
                hidden_states=all_hidden_states if output_hidden_states else None,
                loss=None
            )
        
        # Non-dict return
        return (logits, all_hidden_states) if output_hidden_states else logits
  
