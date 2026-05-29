# artifacts/

Runtime-populated directory. Not committed (except scaffolding and experiment docs).

```
artifacts/
├── datasets/                               # processed HuggingFace datasets (saved to disk)
├── weights/                                # model checkpoints and base model files
└── experiments/                            # experiment descriptions and runtime outputs
    ├── pilot.md                            # HPO pilot: alpha scheduler & LR search
    └── experiment_1_subset_sampling.md     # Exp 1: subset sampling study (AUG)
```
