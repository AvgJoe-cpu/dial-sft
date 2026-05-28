# mdlm_helpers

Runtime helpers for MDLM diffusion training and inference.

## Contents

### `mdlm_scheduler.py`
Defines the noise schedule α(t) used across training and sampling.
- `BaseAlphaScheduler` — abstract base with a self-registering subclass registry; exposes `alpha(t)`, `alpha_derivative(t)`, `weight(t)`, and `reverse_mask_prob(s, t)`
- `LinearAlphaScheduler` — α(t) = 1 − t
- `CosineAlphaScheduler` — α(t) = 1 − cos(π/2 · (1 − t))
- `make_alpha_scheduler(name)` / `get_alpha_scheduler_class(name)` — factory helpers for instantiation by name

### `mdlm_sampler_pt.py`
Unconditional MDLM sampler for pretraining / evaluation. Adapted from [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm).
- `MinimalMDLMSampler` — DDPM reverse sampler with SUBS parameterisation and p(x₀) caching; entry point is `.sample(batch_size, seq_len, num_steps)`

### `mdlm_sampler_sft.py`
Conditional (prompt-conditioned) MDLM sampler for SFT inference.
- `MinimalMDLMSampler` — base sampler, identical to the PT variant but accepts an `attention_mask`
- `SFTMixin` — adds `.sample_sft(prompt_ids, response_length)` for single-row conditional generation
- `SFTMixinBatched` — batched version with BD3LM-style early-exit and NFE counting (`return_nfes=True`)
- `SFTMixinBatchedVarlen` — extends batched sampling to variable-length prompts (list or padded 2-D tensor + `prompt_lens`); handles per-row attention masks and canvas invariant checks

### `mdlm_trainer_pt.py`
HuggingFace `Trainer` subclass for MDLM masked-diffusion pretraining.
- `MDLMConfig` — extends `TrainingArguments` with `time_epsilon` and `loss_weight_type` (`uniform` | `scheduler`)
- `MDLMTrainer` — injects MDLM forward pass (timestep sampling → token masking → weighted cross-entropy); `prediction_step` returns per-token NLL for metric accumulation
- `NLLPPLMetricComputer` — stateful metric accumulator computing NLL and perplexity across eval batches

### `mdlm_trainer_sft.py`
HuggingFace `Trainer` subclass for MDLM SFT, restricting noise and loss to response tokens only.
- `SFTCollator` — pads `input_ids`, `labels`, `attention_mask`, and `assistant_mask` to batch-max length
- `MDLMConfig` — same as PT variant
- `MDLMSFTTrainer` — masks only response positions (via `assistant_mask` or `labels != -100`); enforces `SFTCollator` and right-padding at init time