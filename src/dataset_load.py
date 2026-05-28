from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset, load_dataset, load_from_disk


def count_tokens_in_column_batched(batch, tokenizer=None, column_name: str = "story"):
    tokenized = tokenizer(batch[column_name], truncation=False, padding=False)
    token_counts = [len(input_ids) for input_ids in tokenized["input_ids"]]
    return {f"{column_name}_token_count": token_counts}


def _add_token_counts(ds, tokenizer):
    for col in ("prompt", "story"):
        ds = ds.map(
            count_tokens_in_column_batched,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "column_name": col},
        )
    return ds


def process_and_save_datasets(
    dataset_name: str,
    tokenizer,
    train_save_path: str,
    eval_save_path: str,
    test_save_path: str,
):
    import gc

    print(f"Loading dataset: {dataset_name}")
    dd = load_dataset(dataset_name)
    splits = {
        "train": (dd["train"], train_save_path),
        "validation": (dd["validation"], eval_save_path),
        "test": (dd["test"], test_save_path),
    }
    del dd
    gc.collect()

    for split_name, (ds, save_path) in splits.items():
        print(f"Processing {split_name} split...")
        ds = _add_token_counts(ds, tokenizer)
        print(f"Saving {split_name} split to: {save_path}")
        ds.save_to_disk(save_path)
        del ds
        gc.collect()

    print("✓ All splits processed and saved!")


if __name__ == "__main__":
    TRAIN_OG_PATH = "euclaise/writingprompts"
    TRAIN_OG_SAVE_PATH = "./datasets/base/writingprompts_train"
    EVAL_OG_SAVE_PATH = "./datasets/base/writingprompts_eval"
    TEST_OG_SAVE_PATH = "./datasets/base/writingprompts_test"

    from src.ar.ar_baseline import setup_model_and_tokenizer

    _, tokenizer = setup_model_and_tokenizer(model_name="EleutherAI/pythia-70m")
    process_and_save_datasets(
        dataset_name=TRAIN_OG_PATH,
        tokenizer=tokenizer,
        train_save_path=TRAIN_OG_SAVE_PATH,
        eval_save_path=EVAL_OG_SAVE_PATH,
        test_save_path=TEST_OG_SAVE_PATH,
    )
