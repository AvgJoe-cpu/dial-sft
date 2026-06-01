from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PROJECT_ROOT_ENV_VAR = "DIAL_SFT_PROJECT_ROOT"

_REPO_RELATIVE_ROOTS = {"artifacts", "datasets", "weights", "runs", "conf", "src"}
_LOCAL_MODEL_HINTS = _REPO_RELATIVE_ROOTS | {
    "ar",
    "base",
    "checkpoints",
    "tb_logs",
    "tokenizer",
}


@dataclass(frozen=True)
class ResolvedMdlmPaths:
    train_data: Path
    test_data: Path
    model_load: Path
    model_save: Path
    log_dir: Path


@dataclass(frozen=True)
class ResolvedArPaths:
    train_data_load_path: Path
    train_model_load_path: str
    train_model_save_path: Path
    infer_data_load_path: Path
    infer_model_load_path: str
    infer_data_save_path: Path


class PathResolver:
    def __init__(self, project_root: str | Path | None = None) -> None:
        root_override = project_root or os.environ.get(PROJECT_ROOT_ENV_VAR)
        root_path = (
            Path(root_override).expanduser()
            if root_override is not None
            else Path(__file__).resolve().parents[1]
        )
        self.project_root = root_path.resolve()
        self.artifacts_root = self.project_root / "artifacts"
        self.datasets_root = self.artifacts_root / "datasets"
        self.dataset_base_root = self.datasets_root / "base"
        self.weights_root = self.artifacts_root / "weights"
        self.runs_root = self.artifacts_root / "runs"

    def resolve_project_path(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.project_root / candidate).resolve()

    def resolve_dataset_path(self, value: str | Path) -> Path:
        return self._resolve_category_path(value, self.dataset_base_root)

    def resolve_weights_path(self, value: str | Path) -> Path:
        return self._resolve_category_path(value, self.weights_root)

    def resolve_runs_path(self, value: str | Path) -> Path:
        return self._resolve_category_path(value, self.runs_root)

    def resolve_model_reference(self, value: str | Path) -> str:
        raw_value = str(value)
        if self.is_external_model_ref(raw_value):
            return raw_value
        return str(self.resolve_weights_path(raw_value))

    def tokenizer_cache_dir(self, model_root: str | Path) -> Path:
        return self.resolve_weights_path(model_root) / "tokenizer"

    def tensorboard_log_dir(self, model_root: str | Path) -> Path:
        return self.resolve_weights_path(model_root) / "tb_logs"

    def dataset_subset_path(self, dataset_prefix: str, subset_name: str) -> Path:
        return self.dataset_base_root / f"{dataset_prefix}_{subset_name}"

    def mdlm_checkpoint_files(self, model_root: str | Path) -> dict[str, Path]:
        resolved_root = self.resolve_weights_path(model_root)
        return {
            filename: resolved_root / filename
            for filename in (
                "model.safetensors",
                "modeling_mdlm.py",
                "config.json",
                "configuration_mdlm.py",
            )
        }

    def resolve_mdlm_paths(
        self, paths: object, override_dirname: str
    ) -> ResolvedMdlmPaths:
        return ResolvedMdlmPaths(
            train_data=self.resolve_dataset_path(getattr(paths, "train_data")),
            test_data=self.resolve_dataset_path(getattr(paths, "test_data")),
            model_load=self.resolve_weights_path(getattr(paths, "model_load")),
            model_save=(
                self.resolve_weights_path(getattr(paths, "model_save"))
                / override_dirname
            ),
            log_dir=self.resolve_runs_path(getattr(paths, "log_dir"))
            / override_dirname,
        )

    def resolve_ar_paths(self, paths: object) -> ResolvedArPaths:
        return ResolvedArPaths(
            train_data_load_path=self.resolve_dataset_path(
                getattr(paths, "train_data_load_path")
            ),
            train_model_load_path=self.resolve_model_reference(
                getattr(paths, "train_model_load_path")
            ),
            train_model_save_path=self.resolve_weights_path(
                getattr(paths, "train_model_save_path")
            ),
            infer_data_load_path=self.resolve_dataset_path(
                getattr(paths, "infer_data_load_path")
            ),
            infer_model_load_path=self.resolve_model_reference(
                getattr(paths, "infer_model_load_path")
            ),
            infer_data_save_path=self.resolve_weights_path(
                getattr(paths, "infer_data_save_path")
            ),
        )

    def is_external_model_ref(self, value: str | Path) -> bool:
        text = str(value)
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            return False

        parts = PurePosixPath(text).parts
        if not parts:
            return False
        if parts[0] in {".", "..", "~"} | _REPO_RELATIVE_ROOTS:
            return False
        if len(parts) == 1:
            return False
        return not any(part in _LOCAL_MODEL_HINTS for part in parts)

    def _resolve_category_path(self, value: str | Path, category_root: Path) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        parts = PurePosixPath(str(value)).parts
        if parts and parts[0] in {".", "..", "~"} | _REPO_RELATIVE_ROOTS:
            return self.resolve_project_path(candidate)

        return (category_root / candidate).resolve()
