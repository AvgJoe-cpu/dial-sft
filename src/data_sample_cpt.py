from typing import Optional

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.ipc as ipc
import gc  # added for explicit garbage collection


def sample_clm_dataset(
    target_train_tokens: str = "500M",
    target_eval_tokens:  str = "100M",
    batch_size:          int = 10_000,
    seed:                int = 42,
    accum_batches:       int = 10,
    model_name:          str = "gpt2",
    dataset_name:        str = "HuggingFaceFW/fineweb-edu",
    dataset_config:      str = "sample-350BT",
):
    _units = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000}
    train_tokens = int(target_train_tokens[:-1]) * _units[target_train_tokens[-1]]
    eval_tokens  = int(target_eval_tokens[:-1])  * _units[target_eval_tokens[-1]]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 10**9

    dataset = (
        load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        .shuffle(buffer_size=500_000, seed=seed)
    )

    schema = pa.schema([("text", pa.string()), ("token_count", pa.int32())])
    train_writer = ipc.new_file(pa.OSFile(f"{dataset_name.split('/')[1]}-train-{target_train_tokens}.arrow", "wb"), schema)
    eval_writer  = ipc.new_file(pa.OSFile(f"{dataset_name.split('/')[1]}-eval-{target_eval_tokens}.arrow",  "wb"), schema)

    accum_train = {"text": [], "token_count": []}
    accum_eval  = {"text": [], "token_count": []}
    current_train_tokens = current_eval_tokens = 0

    try:  # added: guarantees writers are closed and resources released even on exception
        for batch in dataset.iter(batch_size=batch_size):
            token_counts = [len(ids) for ids in tokenizer(batch["text"], add_special_tokens=False)["input_ids"]]

            for text, count in zip(batch["text"], token_counts):
                if current_eval_tokens < eval_tokens:
                    accum_eval["text"].append(text);  accum_eval["token_count"].append(count);  current_eval_tokens  += count
                elif current_train_tokens < train_tokens:
                    accum_train["text"].append(text); accum_train["token_count"].append(count); current_train_tokens += count
                else:
                    break

            if current_eval_tokens >= eval_tokens and accum_eval["text"]:
                eval_writer.write_batch(pa.record_batch(accum_eval, schema=schema))
                accum_eval = {"text": [], "token_count": []}

            if len(accum_train["text"]) >= accum_batches * batch_size:
                train_writer.write_batch(pa.record_batch(accum_train, schema=schema))
                accum_train = {"text": [], "token_count": []}

            if current_train_tokens >= train_tokens:
                break

        # final flush inside try so exceptions here are also caught
        if accum_eval["text"]:  eval_writer.write_batch(pa.record_batch(accum_eval,  schema=schema))
        if accum_train["text"]: train_writer.write_batch(pa.record_batch(accum_train, schema=schema))

    finally:  # added: always executes regardless of success or exception
        train_writer.close()
        eval_writer.close()
        del train_writer, eval_writer      # added: release Arrow file handles immediately
        del accum_train, accum_eval        # added: release accumulator memory
        del dataset, tokenizer             # added: release streaming handle and tokenizer weights
        gc.collect()                       # added: force cycle collection — datasets/transformers can hold reference cycles

    # outside try/finally: only prints on clean exit, silent on exception
    print(f"Train: {current_train_tokens:,} tokens")
    print(f"Eval:  {current_eval_tokens:,} tokens")


def post_process(train_path: str, eval_path: str, save_path: Optional[str] = None) -> DatasetDict:
    ds_train = load_dataset("arrow", data_files=train_path)["train"]
    ds_eval  = load_dataset("arrow", data_files=eval_path)["train"]

    dataset_dict = DatasetDict({"train": ds_train, "eval" : ds_eval})  
    if save_path:
        dataset_dict.save_to_disk(save_path)
        print(f"DatasetDict saved to {save_path}")
    
    return dataset_dict

# sample_clm_dataset()
# post_process("fineweb-edu-train-500M.arrow", "fineweb-edu-eval-100M.arrow", save_path="fineweb-edu-sampled")