import gc
import hashlib
import os
import re
from dataclasses import dataclass

from datasets import load_dataset, load_from_disk
from datasets import Dataset


@dataclass
class DatasetProcessingConfig:
    dataset_name: str
    split_save_paths: dict[str, str]


def _process_and_save_datasets(cfg: DatasetProcessingConfig, tokenizer):
    ds_prefix = "".join(re.findall(r'(?:^|_)([a-z])', re.search(r'/(\w+)$', cfg.dataset_name).group(1)))

    print(f"Loading dataset: {cfg.dataset_name}")
    dd = load_dataset(cfg.dataset_name)
    splits = {split: (dd[split], cfg.split_save_paths[split]) for split in dd.keys() & cfg.split_save_paths.keys()}
    del dd
    gc.collect()

    for split_name, (ds, save_path) in splits.items():
        print(f"Processing {split_name} split...")

        def _add_id(batch, indices):
            batch["id"] = [f"{ds_prefix}_{split_name}_{hashlib.sha256(str(idx).encode()).hexdigest()}" for idx in indices]
            return batch
        ds = ds.map(_add_id, batched=True, with_indices=True)

        for old, new in zip((c for c in ds.column_names if c != "id"), ("prompt", "completion")):
            if old != new:
                ds = ds.rename_column(old, new)

        for col in ("prompt", "completion"):
            ds = ds.map(
                lambda batch, col=col: {
                    f"{col}_token_count": [
                        len(ids) for ids in tokenizer(batch[col], truncation=False, padding=False)["input_ids"]
                    ]
                },
                batched=True,
            )

        print(f"Saving {split_name} split to: {save_path}")
        ds.save_to_disk(save_path)
        del ds
        gc.collect()

    print("✓ All splits processed and saved!")



def create_nested_subdatasets(
    D: Dataset,
    sizes: list[int],
    seed: int = 42,
) -> dict[str, Dataset]:
    """
    Build strictly nested sub-datasets D0 ⊂ D1 ⊂ ... ⊂ D_{N-1}
    by one reproducible shuffle of D, then taking prefixes.

    Args:
        D:     full dataset
        sizes: target sizes, sorted strictly ascending; sizes[-1] <= len(D)
        seed:  RNG seed for the one-time shuffle (default 42)

    Returns:
        {"D0": <Dataset>, "D1": <Dataset>, ...}
    """
    # --- validate inputs ---
    if len(sizes) == 0:
        raise ValueError("`sizes` must be non-empty.")
    if any(s <= 0 for s in sizes):
        raise ValueError("All sizes must be positive.")
    if any(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1)):
        raise ValueError("`sizes` must be strictly ascending.")
    if sizes[-1] > len(D):
        raise ValueError(f"largest size {sizes[-1]} exceeds |D|={len(D)}.")

    # --- one-time reproducible shuffle ---
    shuffled = D.shuffle(seed=seed)

    # --- prefix-select per target size ---
    subsets: dict[str, Dataset] = {}
    for i, target_size in enumerate(sizes):
        subsets[f"D{i}"] = shuffled.select(range(target_size))

    return subsets


def verify_nested(subsets: dict[str, Dataset], sizes: list[int]) -> None:
    """Assert nesting, exact sizes, and id-uniqueness using the `id` field."""
    keys = [f"D{i}" for i in range(len(sizes))]

    # 1. exact sizes
    for k, s in zip(keys, sizes):
        assert len(subsets[k]) == s, f"{k}: expected {s}, got {len(subsets[k])}"

    # 2. ids unique within each subset
    for k in keys:
        ids = subsets[k]["id"]
        assert len(set(ids)) == len(ids), f"{k}: duplicate ids found"

    # 3. strict nesting: ids(D_i) ⊆ ids(D_{i+1})
    for a, b in zip(keys[:-1], keys[1:]):
        ids_a = set(subsets[a]["id"])
        ids_b = set(subsets[b]["id"])
        assert ids_a.issubset(ids_b), f"{a} ⊄ {b} (nesting broken)"

    # 4. (stronger) prefix-equality: the first |D_i| rows of D_{i+1} == D_i
    for a, b in zip(keys[:-1], keys[1:]):
        n = len(subsets[a])
        assert subsets[a]["id"] == subsets[b]["id"][:n], (
            f"{a} is not a row-order prefix of {b}"
        )

    print("✓ all checks passed:", {k: len(subsets[k]) for k in keys})

if __name__ == "__main__":
    import os
    cfg = DatasetProcessingConfig(
        dataset_name="euclaise/writingprompts",
        split_save_paths={
            "train":      "./datasets/base/writingprompts_train",
            "validation": "./datasets/base/writingprompts_eval",
            "test":       "./datasets/base/writingprompts_test",
        },
    )

    from src.ar.ar_baseline import setup_model_and_tokenizer
    _, tokenizer = setup_model_and_tokenizer(model_name="EleutherAI/pythia-70m")

    if all(os.path.exists(p) for p in cfg.split_save_paths.values()):
        print("All splits already present on disk, skipping processing.")
    else:
        _process_and_save_datasets(cfg, tokenizer)

    ds = load_from_disk(cfg.split_save_paths["train"])
    print(ds)
    print(ds.column_names)

    ds_prefix = "".join(re.findall(r'(?:^|_)([a-z])', re.search(r'/(\w+)$', cfg.dataset_name).group(1)))
    non_test_paths = [p for k, p in cfg.split_save_paths.items() if k != "test"]
    if len(non_test_paths) > 2:
        from datasets import concatenate_datasets
        train_ds = concatenate_datasets([load_from_disk(p) for p in non_test_paths])
    else:
        train_ds = load_from_disk(non_test_paths[0])

    size_pcts = [0.01, 0.05, 0.20, 1.00]   # D0 ⊂ D1 ⊂ D2 ⊂ D3
    sizes = sorted(set(max(1, round(p * len(train_ds))) for p in size_pcts))

    subsets = create_nested_subdatasets(train_ds, sizes)
    verify_nested(subsets, sizes)

    for key, subset in subsets.items():
        save_path = f"./datasets/base/{ds_prefix}_{key}"
        print(f"Saving {key} ({len(subset)} rows) → {save_path}")
        subset.save_to_disk(save_path)