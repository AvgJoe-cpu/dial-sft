# ar

Autoregressive (AR) model training and inference for story generation. `ar_baseline.py` contains the code for a full train-generate loop.

## Usage

```bash
python3 -m src.ar.ar_baseline
```

This command downloads the training dataset to a temporary directory and runs a single train-inference loop, saving intermediate results to temp storage.
