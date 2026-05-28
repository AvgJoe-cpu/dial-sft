import gc
import hashlib
import re
from dataclasses import dataclass

from datasets import load_dataset, load_from_disk


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


if __name__ == "__main__":
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
    _process_and_save_datasets(cfg, tokenizer)

    ds = load_from_disk(cfg.split_save_paths["train"])
    print(ds)
    print(ds.column_names)