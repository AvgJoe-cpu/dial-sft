from typing import Optional, Callable, Tuple, Union
import torch

def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


class MinimalMDLMSampler:

    def __init__(self, backbone, scheduler, mask_index,
                 time_conditioning=False, neg_infinity=-1_000_000.0):
        self.backbone       = backbone
        self.scheduler      = scheduler
        self.mask_index     = mask_index
        self.time_conditioning = time_conditioning
        self.neg_infinity   = neg_infinity

    # ── forward: raw HF call → subs log-probs ─────────────────────────────────
    def forward(self, x, sigma):
        # sigma is always zeros (time_conditioning=False)
        logits = self.backbone(
            input_ids=x,
            timesteps=sigma,            # model zeros this internally anyway
        ).logits                        # [B, L, V]
        return self._subs_parameterization(logits, x)

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = (xt != self.mask_index)
        logits[unmasked] = self.neg_infinity
        logits[unmasked, xt[unmasked]] = 0
        return logits

    # ── single reverse step ───────────────────────────────────────────────────
    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1

        move_chance_t = (1 - self.scheduler.alpha(t))[:, None, None]
        move_chance_s = (1 - self.scheduler.alpha(t - dt))[:, None, None]

        if p_x0 is None:
            sigma = torch.zeros(x.shape[0], device=x.device)
            p_x0  = self.forward(x, sigma).exp()          # log-probs → probs

        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    # ── outer loop ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, batch_size, seq_len, num_steps=10, eps=1e-5,
               noise_removal=True):
        device = next(self.backbone.parameters()).device

        # prior: all masks
        x = torch.full((batch_size, seq_len), self.mask_index,
                       dtype=torch.long, device=device)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt        = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(batch_size, 1, device=device)
            p_x0_cache, x_next = self._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache)
            if not torch.allclose(x_next, x):   # cache invalid when canvas changes
                p_x0_cache = None
            x = x_next

        if noise_removal:
            sigma = torch.zeros(batch_size, device=device)
            x = self.forward(x, sigma).argmax(dim=-1)

        return x



class SFTMixin:
    """
    Mix into MinimalMDLMSampler:

        class MDLMSampler(SFTMixin, MinimalMDLMSampler):
            pass

    or simply assign:

        MinimalMDLMSampler.sample_sft = SFTMixin.sample_sft
    """

    @torch.no_grad()
    def sample_sft(
        self,
        prompt_ids: torch.LongTensor,    # shape [P] or [1, P]
        response_length: int,
        num_steps: int = 512,
        eps: float = 1e-5,
        noise_removal: bool = True,
    ) -> torch.LongTensor:
        """
        Returns a tensor of shape [1, P + response_length] where the first P
        tokens are bit-identical to `prompt_ids` and the remaining tokens are
        sampled from the diffusion reverse process.
        """
        device = next(self.backbone.parameters()).device

        # --- normalize prompt shape -----------------------------------------
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        assert prompt_ids.ndim == 2 and prompt_ids.shape[0] == 1, (
            "sample_sft handles one prompt per call; got "
            f"shape {tuple(prompt_ids.shape)}"
        )
        prompt_ids = prompt_ids.to(device=device, dtype=torch.long)

        # Guard the SUBS-carry-over invariant: a prompt token that happens to
        # equal mask_index would be (mis)treated as a noised slot.
        assert (prompt_ids != self.mask_index).all(), (
            "prompt contains mask_index; SUBS clamping would treat those "
            "positions as noised."
        )

        P = prompt_ids.shape[1]
        L = P + response_length

        # --- initial canvas: [prompt | MASK ... MASK] -----------------------
        response_init = torch.full(
            (1, response_length), self.mask_index,
            dtype=torch.long, device=device,
        )
        x = torch.cat([prompt_ids, response_init], dim=1)   # [1, L]

        # --- reverse loop (identical structure to .sample) ------------------
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(1, 1, device=device)
            p_x0_cache, x_next = self._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache,
            )
            if not torch.equal(x_next, x):
                p_x0_cache = None
            x = x_next

        if noise_removal:
            sigma = torch.zeros(1, device=device)
            x = self.forward(x, sigma).argmax(dim=-1)

        return x
    


# ----------------------------------------------------------------------------
# Portions adapted from kuleshov-group/mdlm and kuleshov-group/bd3lms,
# both licensed under the Apache License, Version 2.0.
#   - MDLM:   https://github.com/kuleshov-group/mdlm        (Copyright 2024 Cornell University)
#   - BD3LMs: https://github.com/kuleshov-group/bd3lms
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# Specifically borrowed:
#   * the "invalidate cache also when time_conditioning=True" rule from
#     mdlm/diffusion.py::Diffusion._sample (commit c112c52, lines ~682-685)
#   * the "block fully un-masked -> stop forwarding" early-exit idea from
#     bd3lms-family _ddpm_caching_update_ (MBD3LM fork, lines ~1030-1031)
# ----------------------------------------------------------------------------

class SFTMixinBatched:
    """
    Mix into MinimalMDLMSampler:

        class MDLMSampler(SFTMixinBatched, MinimalMDLMSampler):
            pass

    or simply assign:

        MinimalMDLMSampler.sample_sft = SFTMixinBatched.sample_sft
    """

    @torch.no_grad()
    def sample_sft(
        self,
        prompt_ids: torch.LongTensor,        # [P] or [B, P]
        response_length: int,
        num_steps: int = 512,
        eps: float = 1e-5,
        noise_removal: bool = True,
        early_exit: bool = True,             # NEW: BD3LM-style done-check
        return_nfes: bool = False,           # NEW: report NFEs used (cache-hits omitted)
    ) -> Union[torch.LongTensor, Tuple[torch.LongTensor, int]]:
        """
        Batched SFT-style sampling with cache logic borrowed from
        kuleshov-group/mdlm and kuleshov-group/bd3lms (Apache-2.0).

        Accepts:
            prompt_ids of shape [P]    -> treated as B=1
            prompt_ids of shape [B, P] -> batched; all prompts share length P

        Returns (default):  LongTensor [B, P + response_length]
        Returns if return_nfes=True:  (samples, nfes_used)

        Cache logic (vs. the previous version):
          1. Cache invalidation also fires when self.time_conditioning is True,
             matching the upstream MDLM outer-loop rule.
          2. Once every row's RESPONSE region has no MASK tokens left, we
             early-exit the reverse loop -- further calls would be no-ops
             since _ddpm_caching_update gates _x with copy_flag.
          3. Optional prompt-prefix KV warmup via self._prompt_kv_cache_fn,
             called once before the loop. No-op when the hook is absent;
             this is the seam through which BD3LM-style block-KV caching
             can be plugged in by a future backbone without further sampler
             changes.
        """
        device = next(self.backbone.parameters()).device

        # --- normalize prompt shape -----------------------------------------
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        assert prompt_ids.ndim == 2, (
            f"prompt_ids must be 1-D [P] or 2-D [B, P]; got shape "
            f"{tuple(prompt_ids.shape)}"
        )
        prompt_ids = prompt_ids.to(device=device, dtype=torch.long)

        B, P = prompt_ids.shape
        L = P + response_length

        # SUBS carry-over guard -- a prompt token equal to mask_index would
        # be treated by SUBS as a noised slot.
        assert (prompt_ids != self.mask_index).all(), (
            "prompt batch contains mask_index in at least one row; SUBS "
            "clamping would treat those positions as noised."
        )

        # --- initial canvas [B, L] : [prompt | MASK ... MASK] ---------------
        response_init = torch.full(
            (B, response_length), self.mask_index,
            dtype=torch.long, device=device,
        )
        x = torch.cat([prompt_ids, response_init], dim=1)

        # --- optional: prompt-KV warmup (BD3LM-style hook) ------------------
        # If the user wires a callable in (e.g. one that calls
        # backbone(prompt_ids, ..., store_kv=True)), we invoke it once. The
        # parent forward() must then know how to consume that state; this
        # mixin does not assume any particular contract.
        kv_warmup: Optional[Callable] = getattr(self, "_prompt_kv_cache_fn", None)
        if kv_warmup is not None:
            kv_warmup(prompt_ids)

        # --- reverse loop ---------------------------------------------------
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None
        # Matches MDLM outer-loop semantics (default False on this sampler).
        time_conditioning = bool(getattr(self, "time_conditioning", False))

        nfes = 0  # count actual forward calls (a cache hit is NOT an NFE)

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(B, 1, device=device)
            cache_was_hit = p_x0_cache is not None
            p_x0_cache, x_next = self._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache,
            )
            if not cache_was_hit:
                nfes += 1

            # Borrowed from MDLM upstream (Apache-2.0):
            #     if (not torch.allclose(x_next, x) or self.time_conditioning):
            #         p_x0_cache = None
            # We use torch.equal because x is integer-typed; semantically
            # identical here, and avoids the float-tolerance footgun.
            if (not torch.equal(x_next, x)) or time_conditioning:
                p_x0_cache = None
            x = x_next

            # Borrowed from BD3LM family (Apache-2.0):
            # "all tokens in the active block are sampled" -> stop forwarding.
            # In SFT the active block is the response region x[:, P:].
            if early_exit and (x[:, P:] != self.mask_index).all():
                break

        if noise_removal:
            sigma = torch.zeros(B, device=device)
            x = self.forward(x, sigma).argmax(dim=-1)
            nfes += 1

        return (x, nfes) if return_nfes else x


# ============================================================================
# Tests
# ============================================================================
def _build_tiny_sampler(vocab_size=64, mask_index=63, seq_len_cap=64):
    import torch.nn as nn
    from types import SimpleNamespace

    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(vocab_size, 32)
            self.proj = nn.Linear(32, vocab_size)
        def forward(self, input_ids, timesteps=None):
            h = self.emb(input_ids)
            return SimpleNamespace(logits=self.proj(h))

    sampler = MinimalMDLMSampler(   # noqa: F821
        backbone=ToyBackbone().eval(),
        scheduler=LinearAlphaScheduler(),   # noqa: F821
        mask_index=mask_index,
    )
    sampler.sample_sft = SFTMixinBatched.sample_sft.__get__(sampler, type(sampler))
    return sampler, vocab_size, mask_index


def run_sft_sampler_tests():
    """Single-prompt tests -- unchanged from the previous round."""
    torch.manual_seed(0)
    sampler, V, MASK = _build_tiny_sampler()

    prompt = torch.tensor([1, 5, 9, 12, 7, 3], dtype=torch.long)
    P = prompt.shape[0]; R = 10

    out = sampler.sample_sft(prompt, response_length=R, num_steps=32)
    assert out.shape == (1, P + R)
    assert torch.equal(out[0, :P], prompt)
    assert (out != MASK).all()

    torch.manual_seed(1)
    out1 = sampler.sample_sft(prompt, response_length=R, num_steps=32)
    torch.manual_seed(2)
    out2 = sampler.sample_sft(prompt, response_length=R, num_steps=32)
    assert torch.equal(out1[0, :P], prompt) and torch.equal(out2[0, :P], prompt)
    assert not torch.equal(out1[0, P:], out2[0, P:])

    torch.manual_seed(0)
    out_raw = sampler.sample_sft(prompt, response_length=R, num_steps=8,
                                 noise_removal=False)
    assert torch.equal(out_raw[0, :P], prompt)

    print("all single-prompt SFT sampler tests OK")


def run_sft_sampler_batched_tests():
    """Batched tests + cache-borrow regression checks."""
    torch.manual_seed(0)
    sampler, V, MASK = _build_tiny_sampler()

    prompts = torch.tensor(
        [[1, 5, 9, 12, 7, 3],
         [2, 4, 8, 11, 6, 0]],
        dtype=torch.long,
    )
    B, P = prompts.shape; R = 10

    # (a)-(b) shape + per-row prompt preservation
    out = sampler.sample_sft(prompts, response_length=R, num_steps=32)
    assert out.shape == (B, P + R)
    for b in range(B):
        assert torch.equal(out[b, :P], prompts[b])

    # (c) no MASK with noise_removal
    assert (out != MASK).all()

    # (d) per-row independence
    assert not torch.equal(out[0, P:], out[1, P:])

    # (e) noise_removal=False : prompts preserved
    out_raw = sampler.sample_sft(prompts, response_length=R, num_steps=8,
                                 noise_removal=False)
    for b in range(B):
        assert torch.equal(out_raw[b, :P], prompts[b])

    # (f) mask_index guard still trips on batched input
    bad = prompts.clone(); bad[1, 2] = MASK
    try:
        sampler.sample_sft(bad, response_length=R, num_steps=4)
    except AssertionError:
        pass
    else:
        raise AssertionError("guard failed to reject mask_index in a batch row")

    # ---- NEW: borrowed-caching regressions ----

    # (g) NFE accounting: with early_exit=True, NFEs must be <= num_steps + 1
    #     (one extra for noise_removal); with a large num_steps the loop must
    #     terminate well before num_steps once the response is fully sampled.
    torch.manual_seed(0)
    out_big, nfes_big = sampler.sample_sft(
        prompts, response_length=R, num_steps=2048,
        early_exit=True, return_nfes=True,
    )
    assert out_big.shape == (B, P + R)
    assert (out_big != MASK).all()
    # The toy model converges almost immediately. We just assert we did NOT
    # do all 2048 forward passes -- early-exit fired.
    assert nfes_big < 2048, f"early_exit didn't fire: nfes={nfes_big}"
    print(f"  [info] early_exit NFEs (num_steps=2048): {nfes_big}")

    # (h) early_exit=False must do strictly MORE forwards than early_exit=True
    #     for the same seed and steps. (>=, with high probability >.)
    torch.manual_seed(0)
    _, nfes_on  = sampler.sample_sft(prompts, response_length=R,
                                     num_steps=256, early_exit=True,
                                     return_nfes=True)
    torch.manual_seed(0)
    _, nfes_off = sampler.sample_sft(prompts, response_length=R,
                                     num_steps=256, early_exit=False,
                                     return_nfes=True)
    assert nfes_off >= nfes_on, (nfes_off, nfes_on)
    print(f"  [info] NFEs early_exit on/off: {nfes_on}/{nfes_off}")

    # (i) outputs are identical with/without early_exit, given the same seed:
    #     after the early-exit point the loop is a no-op, so the final tensor
    #     and the noise_removal step yield the same answer.
    torch.manual_seed(0)
    a = sampler.sample_sft(prompts, response_length=R, num_steps=256,
                           early_exit=True)
    torch.manual_seed(0)
    b = sampler.sample_sft(prompts, response_length=R, num_steps=256,
                           early_exit=False)
    assert torch.equal(a, b), "early_exit changed the final sample"

    # (j) time_conditioning toggle: when on, the cache must be invalidated
    #     every step -> NFEs == num_steps (+1 for noise removal).
    sampler.time_conditioning = True
    try:
        torch.manual_seed(0)
        _, nfes_tc = sampler.sample_sft(
            prompts, response_length=R, num_steps=8,
            early_exit=False, return_nfes=True,
        )
        # 8 loop steps each force a fresh forward + 1 noise_removal forward
        assert nfes_tc == 8 + 1, f"expected 9 NFEs under time_conditioning, got {nfes_tc}"
    finally:
        sampler.time_conditioning = False

    print("all batched SFT sampler tests OK")


if __name__ == "__main__":
    from src.mdlm.mdlm_helpers.mdlm_scheduler import *
    run_sft_sampler_tests()
    run_sft_sampler_batched_tests()    