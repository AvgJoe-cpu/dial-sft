from __future__ import annotations

from pathlib import Path

from src.paths import PROJECT_ROOT_ENV_VAR, PathResolver

REPO_ROOT = Path(__file__).resolve().parents[1]


class _MdlmPaths:
    train_data = "w_D0"
    test_data = "writingprompts_eval"
    model_load = "base"
    model_save = "checkpoints/stage0"
    log_dir = "pilot/stage0"


class _ArPaths:
    train_data_load_path = "writingprompts_train"
    train_model_load_path = "eleutherai/pythia-70m"
    train_model_save_path = "ar/checkpoints"
    infer_data_load_path = "writingprompts_train"
    infer_model_load_path = "ar/checkpoints"
    infer_data_save_path = "ar/inference_outputs"


def test_path_resolver_discovers_repo_root_without_cwd_dependency(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    resolver = PathResolver()

    assert resolver.project_root == REPO_ROOT


def test_path_resolver_honors_environment_override(monkeypatch, tmp_path: Path) -> None:
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(custom_root))

    resolver = PathResolver()

    assert resolver.project_root == custom_root.resolve()


def test_category_paths_resolve_bare_relative_and_absolute_inputs(
    tmp_path: Path,
) -> None:
    resolver = PathResolver(project_root=tmp_path)
    absolute_target = tmp_path / "absolute" / "dataset"

    assert resolver.resolve_dataset_path("writingprompts_train") == (
        tmp_path / "artifacts" / "datasets" / "base" / "writingprompts_train"
    )
    assert resolver.resolve_dataset_path("artifacts/datasets/base/custom") == (
        tmp_path / "artifacts" / "datasets" / "base" / "custom"
    )
    assert resolver.resolve_dataset_path(absolute_target) == absolute_target.resolve()


def test_model_reference_resolution_preserves_external_refs_and_local_weights() -> None:
    resolver = PathResolver(project_root=REPO_ROOT)

    assert resolver.resolve_model_reference("eleutherai/pythia-70m") == (
        "eleutherai/pythia-70m"
    )
    assert resolver.resolve_model_reference("ar/checkpoints") == str(
        REPO_ROOT / "artifacts" / "weights" / "ar" / "checkpoints"
    )


def test_resolve_mdlm_paths_builds_stage_specific_outputs() -> None:
    resolver = PathResolver(project_root=REPO_ROOT)

    resolved = resolver.resolve_mdlm_paths(_MdlmPaths(), experiment_suffix="lr=1e-4")

    assert resolved.train_data == REPO_ROOT / "artifacts" / "datasets" / "base" / "w_D0"
    assert resolved.model_save == (
        REPO_ROOT / "artifacts" / "weights" / "checkpoints" / "stage0" / "lr=1e-4"
    )
    assert resolved.log_dir == (
        REPO_ROOT / "artifacts" / "runs" / "pilot" / "stage0" / "lr=1e-4"
    )


def test_resolve_ar_paths_and_builders_use_canonical_artifact_layout() -> None:
    resolver = PathResolver(project_root=REPO_ROOT)

    resolved = resolver.resolve_ar_paths(_ArPaths())

    assert resolved.train_data_load_path == (
        REPO_ROOT / "artifacts" / "datasets" / "base" / "writingprompts_train"
    )
    assert resolved.infer_data_save_path == (
        REPO_ROOT / "artifacts" / "weights" / "ar" / "inference_outputs"
    )
    assert resolver.dataset_subset_path("w", "D1") == (
        REPO_ROOT / "artifacts" / "datasets" / "base" / "w_D1"
    )
    assert resolver.tokenizer_cache_dir("ar/checkpoints") == (
        REPO_ROOT / "artifacts" / "weights" / "ar" / "checkpoints" / "tokenizer"
    )
