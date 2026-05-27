from src.mdlm.mdlm_helpers.mdlm_scheduler import LinearAlphaScheduler, CosineAlphaScheduler, BaseAlphaScheduler

from typing import Optional, Any
from dataclasses import dataclass
import torch 
import torch.nn.functional as F 
from transformers import (
    Trainer,
    TrainingArguments
)

@dataclass
class SFTCollator:
    pad_token_id: int

    FILL = {
        "input_ids":      None,    # filled at __post_init__-ish time below
        "labels":         -100,
        "attention_mask": 0,
        "assistant_mask": 0,
    }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        keys = list(features[0].keys())
        max_len = max(len(f["input_ids"]) for f in features)

        # Auto-derive attention_mask if the dataset didn't provide one.
        if "attention_mask" not in keys:
            for f in features:
                f["attention_mask"] = [1] * len(f["input_ids"])
            keys.append("attention_mask")

        fill = dict(self.FILL)
        fill["input_ids"] = self.pad_token_id

        out: dict[str, torch.Tensor] = {}
        for k in keys:
            pad_val = fill.get(k, 0)
            padded = [
                f[k] + [pad_val] * (max_len - len(f[k])) for f in features
            ]
            out[k] = torch.tensor(padded, dtype=torch.long)
        return out

@dataclass
class MDLMConfig(TrainingArguments):
    time_epsilon: float = 0.001
    loss_weight_type: str = "uniform"

    batch_eval_metrics: bool = True
    output_dir: str = "mdlm_output"

    def __post_init__(self):
        super().__post_init__()

        if not (0.0 < self.time_epsilon < 1.0):
            raise ValueError(
                f"time_epsilon must be in (0, 1), got {self.time_epsilon}"
            )
        if self.loss_weight_type not in ("scheduler", "uniform"):
            raise ValueError(
                f"loss_weight_type must be 'scheduler' or 'uniform', "
                f"got {self.loss_weight_type!r}"
            )
        if not self.batch_eval_metrics:
            raise ValueError(
                "MDLMConfig requires batch_eval_metrics=True for per-token "
                "NLL accumulation."
            )
        
class MDLMSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        scheduler: Optional["BaseAlphaScheduler"] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

        cfg: "MDLMConfig" = self.args
        self.scheduler        = scheduler if scheduler is not None else LinearAlphaScheduler()  # noqa: F821
        self.time_epsilon     = cfg.time_epsilon
        self.loss_weight_type = cfg.loss_weight_type

        # --- tokenizer invariants (same as pretraining trainer) -------------
        tok = self.processing_class
        if tok is None:
            raise ValueError("MDLMSFTTrainer requires a tokenizer via `processing_class`.")
        if getattr(tok, "padding_side", None) != "right":
            raise ValueError(f"padding_side must be 'right', got {tok.padding_side!r}.")
        if getattr(tok, "mask_token_id", None) is None:
            raise ValueError("Tokenizer must define `mask_token_id`.")
        if getattr(tok, "pad_token_id", None) is None:
            raise ValueError("Tokenizer must define `pad_token_id` for SFT padding.")

        # --- SFT-specific invariants ----------------------------------------
        if not isinstance(self.data_collator, SFTCollator):  # noqa: F821
            raise ValueError(
                "MDLMSFTTrainer requires `data_collator=SFTCollator(...)`. "
                f"Got {type(self.data_collator).__name__}. A pretraining "
                "collator would pad `labels` with values != -100 and pollute "
                "the maskable set."
            )
        if self.data_collator.pad_token_id != tok.pad_token_id:
            raise ValueError(
                f"SFTCollator.pad_token_id ({self.data_collator.pad_token_id}) "
                f"!= tokenizer.pad_token_id ({tok.pad_token_id})."
            )

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs, _, _ = self._sft_forward(model, inputs)
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, _, token_nll, maskable_mask = self._sft_forward(model, inputs)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        predictions = token_nll.detach().contiguous()
        label_ids   = maskable_mask.to(predictions.dtype).detach().contiguous()
        return (loss.detach(), predictions, label_ids)

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "MDLMSFTTrainer does not support predict(); use a sampler "
            "(see MinimalMDLMSampler.sample_sft) for generation."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _derive_maskable_mask(inputs: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Prefer the explicit `assistant_mask` carried by the SFT pipeline;
        fall back to `labels != -100` (equivalent post-collation).
        """
        if "assistant_mask" in inputs:
            return inputs["assistant_mask"].bool()
        return labels != -100

    def _sft_forward(self, model, inputs):
        input_ids      = inputs["input_ids"]
        labels         = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        b, l           = input_ids.shape

        maskable_mask = self._derive_maskable_mask(inputs, labels)

        # 1. timesteps
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # 2. noise — restricted to response tokens by maskable_mask
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # 3. forward — attention_mask matters in SFT because of right-padding
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)

        # 4. per-row weights
        loss_weights = (
            self.scheduler.weight(t).unsqueeze(1)
            if self.loss_weight_type == "scheduler"
            else 1.0
        )

        # 5. weighted CE — scored only where we noised (subset of response)
        # Invariant from the data pipeline; cheap to assert in training.
        assert (input_ids[maskable_mask] == labels[maskable_mask]).all(), \
            "input_ids and labels disagree at response positions"
        token_nll = F.cross_entropy(
            outputs.logits.transpose(1, 2), input_ids, reduction="none",
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

        # 6. denominator = response tokens in batch (NOT noised count; that
        #    would inflate loss on rows where few tokens happened to be noised).
        loss = token_nll.sum() / maskable_mask.sum().clamp_min(1)

        return loss, outputs, token_nll, maskable_mask



# HAS TEST 
def run_sft_trainer_smoke():
    import torch.nn as nn
    from types import SimpleNamespace
    from datasets import Dataset
    import tempfile

    torch.manual_seed(0)

    tok = SimpleNamespace(padding_side="right", mask_token_id=63, pad_token_id=0)

    VOCAB = 64
    class ToyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(VOCAB, 16)
            self.proj = nn.Linear(16, VOCAB)
        def forward(self, input_ids, attention_mask=None, **_):
            return SimpleNamespace(logits=self.proj(self.emb(input_ids)))

    model   = ToyLM()
    rows    = [
        {"input_ids": [1,10,11,12,20,21,2],       "labels": [-100,-100,-100,-100,20,21,-100],       "assistant_mask": [0,0,0,0,1,1,0]},
        {"input_ids": [1,10,11,30,31,32,33,2],     "labels": [-100,-100,-100,30,31,32,33,-100],      "assistant_mask": [0,0,0,1,1,1,1,0]},
        {"input_ids": [1,10,11,12,13,40,2],        "labels": [-100,-100,-100,-100,-100,40,-100],     "assistant_mask": [0,0,0,0,0,1,0]},
    ]
    collator = SFTCollator(pad_token_id=tok.pad_token_id)

    with tempfile.TemporaryDirectory() as tmp:
        args = MDLMConfig(
            output_dir=tmp,
            num_train_epochs=1,
            per_device_train_batch_size=3,
            learning_rate=1e-3,
            logging_steps=1,
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            batch_eval_metrics=True,
            remove_unused_columns=False,
        )

        def build(col):
            return MDLMSFTTrainer(
                model=model, args=args, train_dataset=Dataset.from_list(rows * 8),
                processing_class=tok, data_collator=col,
            )

        # (a) training runs end-to-end
        build(collator).train()

        # (b) one forward → finite loss, correct shapes
        batch = {k: v.to(next(model.parameters()).device) for k, v in collator(rows).items()}
        loss, _, token_nll, maskable_mask = build(collator)._sft_forward(model, batch)
        assert torch.isfinite(loss),                        f"non-finite loss: {loss}"
        assert token_nll.shape == batch["input_ids"].shape
        assert maskable_mask.dtype == torch.bool

        # (c) wrong collator type → rejected
        try:    build(SimpleNamespace(pad_token_id=tok.pad_token_id, __call__=collator))
        except ValueError as e: assert "SFTCollator" in str(e)
        else:   raise AssertionError("non-SFTCollator was accepted")

        # (d) mismatched pad_token_id → rejected
        try:    build(SFTCollator(pad_token_id=tok.pad_token_id + 1))
        except ValueError as e: assert "pad_token_id" in str(e)
        else:   raise AssertionError("mismatched pad_token_id was accepted")

    print(f"  final-step loss : {loss.item():.4f}")
    print(f"  response tokens : {int(maskable_mask.sum().item())}")
if __name__ == "__main__":
    run_sft_trainer_smoke()