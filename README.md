# dial-sft

## Contents

```
dial-sft/
├── src/                  # Source code
│   ├── ar/               # Autoregressive baseline
│   ├── mdlm/             # MDLM model, trainers, samplers
│   └── dataset_load.py   # Dataset loading utilities
├── datasets/             # Raw and processed datasets
└── weights/              # Model checkpoints and weights
```


## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup (first time after clone)

```bash
make setup
```

This installs all dependencies and registers the git hooks. From this point, code quality checks (formatting, linting, type-checking) run automatically on every `git commit`.

### Manual commands

```bash
make lint       # run all linters across the full codebase
make typecheck  # run mypy type checker
uv sync         # install runtime deps only
uv sync --group dev  # install runtime + dev deps
```

### Run scripts

Prefix commands with `uv run` to execute within the managed environment:

```bash
uv run python hydra_mdlm_sft.py
uv run pytest
uv run black src/
```

> **Note:** After merging, run `uv lock` locally to generate the `uv.lock` file and commit it to the repository. This ensures fully reproducible installs across machines and CI. Run `uv lock --upgrade` to update all dependencies.
