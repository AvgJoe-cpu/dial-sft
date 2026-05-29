import math
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mdlm.mdlm_helpers.mdlm_scheduler import (
    CosineAlphaScheduler,
    LinearAlphaScheduler,
)
from src.mdlm.mdlm_helpers.mdlm_trainer_pt import MDLMConfig as PTConfig
from src.mdlm.mdlm_helpers.mdlm_trainer_pt import MDLMTrainer
from src.mdlm.mdlm_helpers.mdlm_trainer_sft import MDLMConfig as SFTConfig
from src.mdlm.mdlm_helpers.mdlm_trainer_sft import MDLMSFTTrainer, SFTCollator

VOCAB = 16
SEQ = 6
BATCH = 2
HIDDEN = 16
SCALE = 4.0
MASK_TOKEN_ID = 0
PAD_TOKEN_ID = 1
TIME_EPSILON = 0.001
MASK_SEED = 0


class MockTokenizer:
    mask_token_id = MASK_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    padding_side = "right"


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(SCALE * torch.eye(VOCAB))

    def forward(self, input_ids, attention_mask=None):
        x = F.one_hot(input_ids, num_classes=HIDDEN).float()
        logits = self.linear(x)
        return SimpleNamespace(logits=logits)


class FixedTLinearScheduler:
    """Linear alpha scheduler with externally fixed per-row t values."""

    def __init__(self, fixed_t: torch.Tensor):
        self.fixed_t = fixed_t.float()

    def _expand(self, t: torch.Tensor) -> torch.Tensor:
        out = self.fixed_t.to(t.device)
        if out.ndim == 0:
            out = out.expand_as(t)
        return out

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        fixed_t = self._expand(t)
        return 1.0 - fixed_t

    def weight(self, t: torch.Tensor) -> torch.Tensor:
        fixed_t = self._expand(t)
        return 1.0 / (fixed_t + 1e-6)


def make_batch(all_response: bool = False):
    input_ids = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    if all_response:
        labels = input_ids.clone()
    else:
        labels = torch.tensor(
            [
                [-100, -100, 4, 5, 6, -100],
                [-100, -100, -100, 6, 7, 8],
            ],
            dtype=torch.long,
        )
    assistant_mask = (labels != -100).long()
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    return input_ids, labels, assistant_mask, attention_mask


def make_toy_trainer(
    model, loss_weight_type: str, scheduler, trainer_kind: str = "sft"
):
    output_dir = tempfile.mkdtemp(prefix="mdlm-validate-")
    common_kwargs = dict(
        output_dir=output_dir,
        use_cpu=True,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        batch_eval_metrics=True,
        time_epsilon=TIME_EPSILON,
        loss_weight_type=loss_weight_type,
    )

    if trainer_kind == "sft":
        args = SFTConfig(**common_kwargs)
        return MDLMSFTTrainer(
            model=model,
            args=args,
            train_dataset=[],
            processing_class=MockTokenizer(),
            data_collator=SFTCollator(pad_token_id=PAD_TOKEN_ID),
            scheduler=scheduler,
        )

    args = PTConfig(**common_kwargs)
    return MDLMTrainer(
        model=model,
        args=args,
        train_dataset=[],
        processing_class=MockTokenizer(),
        data_collator=SFTCollator(pad_token_id=PAD_TOKEN_ID),
        scheduler=scheduler,
    )


def ground_truth_forward(
    input_ids,
    labels,
    assistant_mask,
    attention_mask,
    t,
    scheduler,
    loss_weight_type,
    align_with_trainer_rand_order=False,
):
    del attention_mask
    b, l = input_ids.shape

    maskable_mask = assistant_mask.bool()
    alpha_t = scheduler(t)
    p_mask = 1.0 - alpha_t
    p_mask_expanded = p_mask.unsqueeze(1).expand(b, l)

    torch.manual_seed(MASK_SEED)
    if align_with_trainer_rand_order:
        _ = torch.rand((b,))
    rand_draw = torch.rand((b, l))
    masked_mask = (rand_draw < p_mask_expanded) & maskable_mask

    noised_input_ids = torch.where(masked_mask, MASK_TOKEN_ID, input_ids)

    logits = SCALE * F.one_hot(noised_input_ids, num_classes=VOCAB).float()
    z = math.exp(SCALE) + VOCAB - 1
    ce_per_position = torch.full((b, l), math.log(z), dtype=torch.float32)
    ce_per_position = ce_per_position - SCALE * (noised_input_ids == input_ids).float()

    if loss_weight_type == "uniform":
        loss_weights = 1.0
    else:
        loss_weights = scheduler.weight(t).unsqueeze(1)

    token_nll = ce_per_position * loss_weights * masked_mask.float()
    loss = token_nll.sum() / maskable_mask.sum().clamp_min(1)

    return SimpleNamespace(
        maskable_mask=maskable_mask,
        p_mask=p_mask,
        masked_mask=masked_mask,
        noised_input_ids=noised_input_ids,
        logits=logits,
        token_nll=token_nll,
        loss=loss,
        alpha_t=alpha_t,
        loss_weights=loss_weights,
    )


def check_1_scheduler_correctness():
    grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    lin = LinearAlphaScheduler()
    cos = CosineAlphaScheduler()

    expected_lin = 1.0 - grid
    got_lin = lin(grid)
    assert torch.equal(
        got_lin, expected_lin
    ), f"linear alpha mismatch: {got_lin} vs {expected_lin}"

    expected_cos = 1.0 - torch.cos((math.pi / 2) * (1.0 - grid))
    got_cos = cos(grid)
    assert torch.allclose(
        got_cos, expected_cos, atol=1e-6
    ), f"cosine alpha mismatch: {got_cos} vs {expected_cos}"

    weight_grid = torch.tensor([0.1, 0.3, 0.7])
    expected_weight = 1.0 / (weight_grid + 1e-6)
    got_weight = lin.weight(weight_grid)
    assert torch.allclose(
        got_weight, expected_weight, atol=1e-5
    ), f"linear weight mismatch: {got_weight} vs {expected_weight}"

    return True, ""


def check_2_masking_invariants():
    input_ids, labels, assistant_mask, attention_mask = make_batch()
    t = torch.tensor([0.3, 0.7])
    gt = ground_truth_forward(
        input_ids,
        labels,
        assistant_mask,
        attention_mask,
        t,
        FixedTLinearScheduler(t),
        "uniform",
    )

    assert (
        (gt.masked_mask & ~gt.maskable_mask).sum() == 0
    ).item(), "masked prompt token found"
    assert (
        gt.masked_mask.sum() < gt.maskable_mask.sum()
    ).item(), f"masked_mask is not strict subset ({gt.masked_mask.sum()} vs {gt.maskable_mask.sum()})"
    assert torch.equal(
        gt.noised_input_ids[~gt.masked_mask], input_ids[~gt.masked_mask]
    ), "non-masked positions changed"
    assert torch.equal(
        gt.noised_input_ids[gt.masked_mask],
        torch.full_like(gt.noised_input_ids[gt.masked_mask], MASK_TOKEN_ID),
    ), "masked positions are not all MASK_TOKEN_ID"

    return True, ""


def check_3_trainer_masking_matches_ground_truth():
    input_ids, labels, assistant_mask, attention_mask = make_batch()
    t = torch.tensor([0.3, 0.7])
    scheduler = FixedTLinearScheduler(t)
    gt = ground_truth_forward(
        input_ids,
        labels,
        assistant_mask,
        attention_mask,
        t,
        scheduler,
        "uniform",
        align_with_trainer_rand_order=True,
    )

    model = ToyModel()
    trainer = make_toy_trainer(model, "uniform", scheduler, trainer_kind="sft")
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "attention_mask": attention_mask,
    }
    torch.manual_seed(MASK_SEED)
    _, _, token_nll, _ = trainer._sft_forward(model, inputs)
    masked_mask_from_trainer = token_nll > 0

    assert torch.equal(
        masked_mask_from_trainer, gt.masked_mask
    ), f"masked mask mismatch\ntrainer={masked_mask_from_trainer}\ngt={gt.masked_mask}"
    return True, ""


def check_4_loss_numerator_token_nll():
    input_ids, labels, assistant_mask, attention_mask = make_batch()
    t = torch.tensor([0.3, 0.7])
    scheduler = FixedTLinearScheduler(t)
    gt = ground_truth_forward(
        input_ids,
        labels,
        assistant_mask,
        attention_mask,
        t,
        scheduler,
        "uniform",
        align_with_trainer_rand_order=True,
    )

    model = ToyModel()
    trainer = make_toy_trainer(model, "uniform", scheduler, trainer_kind="sft")
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "attention_mask": attention_mask,
    }
    torch.manual_seed(MASK_SEED)
    _, _, token_nll, _ = trainer._sft_forward(model, inputs)

    assert torch.allclose(
        token_nll, gt.token_nll, atol=1e-5
    ), f"token_nll mismatch\ntrainer={token_nll}\ngt={gt.token_nll}"
    if (~gt.masked_mask).any():
        assert (
            token_nll[~gt.masked_mask].abs().max().item() < 1e-9
        ), "non-masked token_nll not zero"
    return True, ""


def _run_denominator_scenario(t: torch.Tensor):
    input_ids, labels, assistant_mask, attention_mask = make_batch()
    scheduler = FixedTLinearScheduler(t)
    gt = ground_truth_forward(
        input_ids,
        labels,
        assistant_mask,
        attention_mask,
        t,
        scheduler,
        "uniform",
        align_with_trainer_rand_order=True,
    )

    model = ToyModel()
    trainer = make_toy_trainer(model, "uniform", scheduler, trainer_kind="sft")
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "attention_mask": attention_mask,
    }
    torch.manual_seed(MASK_SEED)
    loss, _, _, _ = trainer._sft_forward(model, inputs)

    correct_denom = gt.maskable_mask.sum().clamp_min(1)
    expected_loss = gt.token_nll.sum() / correct_denom
    wrong_loss = gt.token_nll.sum() / gt.masked_mask.sum().clamp_min(1)

    return SimpleNamespace(
        loss=loss,
        gt=gt,
        expected_loss=expected_loss,
        wrong_loss=wrong_loss,
        correct_denom=correct_denom,
    )


def check_5_loss_denominator_ddp_critical():
    # Critical DDP regression check: if loss is divided by masked_mask.sum() instead
    # of maskable_mask.sum(), low-mask-rate batches produce inflated loss and diverge
    # across ranks where masked counts differ.
    scenario_a = _run_denominator_scenario(torch.tensor([0.3, 0.7]))
    scenario_b = _run_denominator_scenario(torch.tensor([0.05, 0.05]))

    assert (
        scenario_a.correct_denom.item() == 6
    ), f"expected denominator 6, got {scenario_a.correct_denom.item()}"
    assert torch.allclose(
        scenario_a.loss, scenario_a.expected_loss, atol=1e-5
    ), f"Scenario A loss mismatch: trainer={scenario_a.loss} expected={scenario_a.expected_loss}"

    assert torch.allclose(
        scenario_b.loss, scenario_b.expected_loss, atol=1e-5
    ), f"Scenario B loss mismatch: trainer={scenario_b.loss} expected={scenario_b.expected_loss}"
    assert (
        scenario_b.gt.masked_mask.sum().item() < scenario_b.correct_denom.item()
    ), "Scenario B did not produce sparse masking; test is not discriminating"
    assert not torch.allclose(
        scenario_b.wrong_loss, scenario_b.expected_loss, atol=1e-6
    ), "Test is not discriminating — masked_mask.sum() == maskable_mask.sum() in this scenario"

    details = [
        f"✓  correct denominator: maskable_mask.sum() = {scenario_a.correct_denom.item()}",
        "✓  Scenario B (low t): loss matches correct denominator",
        "✓  wrong denominator gives different result (check discriminates)",
    ]
    return True, "\n".join(details)


def check_6_loss_weight_modes():
    input_ids, labels, assistant_mask, attention_mask = make_batch()
    t = torch.tensor([0.3, 0.7])
    scheduler = FixedTLinearScheduler(t)
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "attention_mask": attention_mask,
    }

    model_u = ToyModel()
    trainer_u = make_toy_trainer(model_u, "uniform", scheduler, trainer_kind="sft")
    gt_u = ground_truth_forward(
        input_ids,
        labels,
        assistant_mask,
        attention_mask,
        t,
        scheduler,
        "uniform",
        align_with_trainer_rand_order=True,
    )
    torch.manual_seed(MASK_SEED)
    loss_u, _, _, _ = trainer_u._sft_forward(model_u, inputs)

    model_s = ToyModel()
    trainer_s = make_toy_trainer(model_s, "scheduler", scheduler, trainer_kind="sft")
    gt_s = ground_truth_forward(
        input_ids,
        labels,
        assistant_mask,
        attention_mask,
        t,
        scheduler,
        "scheduler",
        align_with_trainer_rand_order=True,
    )
    torch.manual_seed(MASK_SEED)
    loss_s, _, _, _ = trainer_s._sft_forward(model_s, inputs)

    denom = gt_u.maskable_mask.sum().clamp_min(1)
    expected_uniform = gt_u.token_nll.sum() / denom

    assert torch.allclose(
        loss_u, expected_uniform, atol=1e-5
    ), f"uniform loss mismatch: {loss_u} vs {expected_uniform}"
    assert not torch.allclose(
        loss_s, loss_u, atol=1e-6
    ), "scheduler loss should differ from uniform loss"
    assert torch.allclose(
        loss_s, gt_s.loss, atol=1e-5
    ), f"scheduler loss mismatch: {loss_s} vs {gt_s.loss}"

    return True, ""


def check_7_backward_gradients():
    input_ids, labels, assistant_mask, attention_mask = make_batch()
    t = torch.tensor([0.3, 0.7])
    scheduler = FixedTLinearScheduler(t)
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "attention_mask": attention_mask,
    }

    model = ToyModel()
    trainer = make_toy_trainer(model, "uniform", scheduler, trainer_kind="sft")
    gt = ground_truth_forward(
        input_ids,
        labels,
        assistant_mask,
        attention_mask,
        t,
        scheduler,
        "uniform",
        align_with_trainer_rand_order=True,
    )

    torch.manual_seed(MASK_SEED)
    loss, _, _, _ = trainer._sft_forward(model, inputs)
    loss.backward()

    grad = model.linear.weight.grad
    assert loss.requires_grad, "loss should require gradients"
    assert grad is not None, "gradient is None"
    assert grad.abs().sum().item() > 0, "gradient is all zeros"

    expected_grad = torch.zeros_like(grad)
    denom = gt.maskable_mask.sum().clamp_min(1).float()
    z = math.exp(SCALE) + VOCAB - 1
    probs = torch.full((VOCAB,), 1.0 / z)
    probs[0] = math.exp(SCALE) / z

    masked_positions = torch.nonzero(gt.masked_mask, as_tuple=False)
    for pos in masked_positions:
        b_idx, l_idx = int(pos[0]), int(pos[1])
        target = int(input_ids[b_idx, l_idx])
        contrib = probs.clone()
        contrib[target] -= 1.0
        expected_grad[:, MASK_TOKEN_ID] += contrib / denom

    assert torch.allclose(
        grad, expected_grad, atol=1e-6
    ), f"gradient mismatch\nactual={grad}\nexpected={expected_grad}"

    first_target = int(input_ids[masked_positions[0, 0], masked_positions[0, 1]])
    softmax_target = probs[first_target].item()
    single_token_contrib = probs.clone()
    single_token_contrib[first_target] -= 1.0
    single_token_contrib = single_token_contrib / denom
    expected_correct = -(1.0 - softmax_target) / denom.item()
    assert math.isclose(
        single_token_contrib[first_target].item(), expected_correct, rel_tol=1e-6
    ), "single-token analytical CE gradient mismatch"
    assert (
        grad[first_target, first_target].abs().item() < 1e-9
    ), "gradient should not flow through original-token diagonal when noised token is MASK"
    assert (
        grad[first_target, MASK_TOKEN_ID].abs().item() > 0.0
    ), "correct-class gradient should flow through MASK-token input column"

    assert (
        grad[:, 1:].abs().max().item() < 1e-9
    ), "gradient should be zero for input-token columns never used by noised masked inputs"

    return True, ""


def check_8_sft_vs_pt_consistency():
    input_ids, labels, _, attention_mask = make_batch(all_response=True)
    assistant_mask = torch.ones_like(input_ids)
    t = torch.tensor([0.3, 0.7])
    scheduler = FixedTLinearScheduler(t)

    sft_inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "attention_mask": attention_mask,
    }
    pt_inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    model_s = ToyModel()
    trainer_s = make_toy_trainer(model_s, "uniform", scheduler, trainer_kind="sft")
    torch.manual_seed(MASK_SEED)
    loss_s, _, _, _ = trainer_s._sft_forward(model_s, sft_inputs)

    model_p = ToyModel()
    trainer_p = make_toy_trainer(model_p, "uniform", scheduler, trainer_kind="pt")
    torch.manual_seed(MASK_SEED)
    loss_p, _, _, _ = trainer_p._mdlm_forward(model_p, pt_inputs)

    assert torch.allclose(
        loss_s, loss_p, atol=1e-5
    ), f"SFT/PT loss mismatch: sft={loss_s}, pt={loss_p}"

    return True, ""


def run_check(index: int, title: str, fn):
    try:
        passed, details = fn()
        if not passed:
            raise AssertionError(details)
        print(f"  Check {index}  {title:.<34} PASS")
        if details:
            for line in details.split("\n"):
                print(f"    {line}")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Check {index}  {title:.<34} FAIL")
        print(f"    {exc}")
        return False


def main():
    print("═" * 54)
    print("  MDLM Forward Pass Validation")
    print("═" * 54)
    print()

    checks = [
        ("Scheduler correctness", check_1_scheduler_correctness),
        ("Masking invariants", check_2_masking_invariants),
        (
            "Trainer masking vs ground truth",
            check_3_trainer_masking_matches_ground_truth,
        ),
        ("Loss numerator (token_nll)", check_4_loss_numerator_token_nll),
        ("Loss denominator (DDP-critical)", check_5_loss_denominator_ddp_critical),
        ("Loss weight modes", check_6_loss_weight_modes),
        ("Backward pass gradients", check_7_backward_gradients),
        ("SFT vs PT trainer consistency", check_8_sft_vs_pt_consistency),
    ]

    passed = 0
    for idx, (title, fn) in enumerate(checks, start=1):
        passed += int(run_check(idx, title, fn))

    print()
    print("═" * 54)
    print(f"  Result: {passed}/{len(checks)} PASSED")
    print("═" * 54)

    if passed != len(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
