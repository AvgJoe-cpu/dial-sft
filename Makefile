.PHONY: setup lint typecheck

setup: ## One-time setup after clone: install deps and git hooks
	uv sync --group dev
	uv run pre-commit install

lint: ## Run all linters across the full codebase
	uv run pre-commit run --all-files

typecheck: ## Run mypy type checker
	uv run mypy src/ src/hydra_mdlm_sft.py
