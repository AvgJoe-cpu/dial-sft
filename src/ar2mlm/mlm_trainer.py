import math
import torch
import torch.nn.functional as F
import transformers


class MDLMTrainer(transformers.Trainer):
    def __init__(
        self,
        # args: MDLMConfig,                                # removed - plain TrainingArguments passed directly
        *args,
        scheduler=None,
        time_epsilon: float = 1e-3,                        # was: self.time_epsilon = args.time_epsilon
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # if not (0.0 < args.time_epsilon < 1.0):          # removed - validation dropped
        #     raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler    = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = time_epsilon

        # self.loss_weight_type = args.loss_weight_type    # removed - hardcoded to scheduler
        # self.loss_norm_type = args.loss_norm_type        # removed - hardcoded to token
        # self.right_shift_logits = args.right_shift_logits  # removed - not needed post-A2D

        # self.meter = OnEvaluateMetricsCallback(...)      # removed - HF Trainer logs loss natively
        # self.add_callback(self.meter)                    # removed

    # def _preprocess_inputs(self, inputs):                # removed - only existed for right_shift_logits
    #     if self.right_shift_logits:
    #         ...prepend_bos(...)
    #     return inputs

    # def _postprocess_outputs(self, outputs):             # removed - only existed for right_shift_logits
    #     if self.right_shift_logits:
    #         outputs.logits = torch.cat(...)
    #     return outputs

    # def _compute_loss_weights(self, t, inputs, ...):     # removed - one line, not worth a method
    #     if self.loss_weight_type == "scheduler": ...
    #     elif self.loss_weight_type == "uniform": ...

    # @torch.no_grad()                                     # removed
    # def prediction_step(...):                            # removed - Trainer default is sufficient
    #     ...

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # assert self.processing_class.padding_side == "right"  # removed - move to data sanity check
        # inputs = self._preprocess_inputs(inputs)         # removed - method dropped

        input_ids      = inputs["input_ids"]
        labels         = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        b, l          = input_ids.shape
        maskable_mask = labels != -100                     # [b, l]

        # 1. sample t ~ Uniform[ε, 1)
        t      = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=input_ids.device)
        p_mask = 1.0 - self.scheduler.alpha(t).unsqueeze(1).expand(b, l)  # was: self.scheduler(t)

        # 2. stochastic masking — maskable positions only
        masked_mask      = (torch.rand((b, l), device=input_ids.device) < p_mask) & maskable_mask
        noised_input_ids = torch.where(masked_mask, self.processing_class.mask_token_id, input_ids)

        # 3. forward
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        # outputs = self._postprocess_outputs(outputs)     # removed - method dropped
        logits  = outputs.logits                           # [b, l, V]

        # 4. loss weights w(t) broadcast over sequence
        # was: self._compute_loss_weights(t=t, inputs=inputs, masked_mask=masked_mask)
        loss_weights = self.scheduler.weight(t).unsqueeze(1).expand(b, l)

        # 5. weighted cross-entropy
        # assert (input_ids[maskable_mask] == labels[maskable_mask]).all()  # removed - move to data check
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")  # [b, l]
        token_nll = token_nll * loss_weights * masked_mask.float()

        # self.meter.update(...)                           # removed - meter dropped

        # 6. normalize by total maskable tokens
        # was: if self.loss_norm_type == "token" / "sequence" / "batch" branching
        loss = token_nll.sum() / maskable_mask.sum().clamp_min(1)

        return (loss, outputs) if return_outputs else loss