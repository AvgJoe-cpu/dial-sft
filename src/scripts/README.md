# Scripts

Thin CLI wrappers around core package logic. All scripts use [Typer](https://typer.tiangolo.com/) — run with no arguments for an interactive prompt, or pass flags directly.

---

## `process_dataset.py` — Dataset Preprocessing

Loads a HuggingFace dataset, assigns IDs, renames columns to `prompt`/`completion`, computes token counts, and saves splits to disk.

```bash
process-dataset                        # interactive wizard
process-dataset --help                 # show all options
process-dataset \
  --dataset-name euclaise/writingprompts \
  --model-name EleutherAI/pythia-70m \
  --output-dir ./datasets/base
```

---

## `mdlm_train_sft.py` — MDLM SFT Training

Loads preprocessed splits from disk, applies chat templating, and runs MDLM supervised fine-tuning.

```bash
mdlm-train-sft                         # interactive wizard
mdlm-train-sft --help                  # show all options
mdlm-train-sft \
  --train-data ./datasets/base/writingprompts_train \
  --test-data  ./datasets/base/writingprompts_test \
  --model-load ./weights/base \
  --model-save ./weights/checkpoints \
  --num-epochs 2 \
  --batch-size 16
```

---

> Run `process-dataset` before `mdlm-train-sft` — the training script expects data already saved to disk.