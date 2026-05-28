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

└── dataset_load.py           # Dataset loading utilities
```

