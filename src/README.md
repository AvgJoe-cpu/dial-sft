# src

Source code for `dial-sft`.

## Contents

```
src/
├── ar/
│   └── ar_baseline.py        # Autoregressive baseline
├── mdlm/
│   ├── modeling_mdlm.py      # MDLM model definition
│   ├── load_model.py         # Model loading utilities
│   ├── mdlm_train_sft.py     # MDLM SFT training
│   ├── mdlm_inference.py     # MDLM inference
│   └── mdlm_helpers/
│       ├── mdlm_sampler_pt.py    # Pretraining sampler
│       ├── mdlm_sampler_sft.py   # SFT sampler
│       ├── mdlm_scheduler.py     # Noise scheduler
│       ├── mdlm_trainer_pt.py    # Pretraining trainer
│       └── mdlm_trainer_sft.py   # SFT trainer
├── scripts/
│   ├── process_dataset.py    # CLI: dataset preprocessing
│   ├── mdlm_train_sft.py     # CLI: MDLM SFT training
│   └── README.md             # Scripts usage guide
└── dataset_load.py           # Dataset loading utilities
```

## Scripts

CLI entry points live in `src/scripts/` — thin Typer wrappers around the core logic. See [`scripts/README.md`](scripts/README.md) for usage.

| Command | Script | Wraps |
|---|---|---|
| `process-dataset` | `scripts/process_dataset.py` | `dataset_load.py` |
| `mdlm-train-sft` | `scripts/mdlm_train_sft.py` | `mdlm/mdlm_train_sft.py` |