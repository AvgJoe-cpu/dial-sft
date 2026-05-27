from src.mdlm.load_model import load_model
from src.mdlm.mdlm_helpers.mdlm_trainer_sft import *
from src.mdlm.mdlm_helpers.mdlm_scheduler import *
from src.mdlm.mdlm_helpers.mdlm_sampler_sft import * 

from datasets import Dataset

def _sft_map_fn(example):
    enc = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids      = enc["input_ids"]
    assistant_mask = enc["assistant_masks"]
    labels = [tok if m == 1 else -100
              for tok, m in zip(input_ids, assistant_mask)]
    return {"input_ids": input_ids, "labels": labels, "assistant_mask": assistant_mask}



if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:

        model, tokenizer = load_model(local_dir=tmp)

        TOY_ROWS = [
            {"prompt": "What is 2 + 2?",                   "completion": "4"},
            {"prompt": "Name a primary color.",             "completion": "Blue"},
            {"prompt": "Capital of France?",                "completion": "Paris"},
            {"prompt": "Say hello.",                        "completion": "Hello!"},
            {"prompt": "Largest planet?",                   "completion": "Jupiter"},
            {"prompt": "Opposite of hot?",                  "completion": "Cold"},
            {"prompt": "How many legs does a spider have?", "completion": "Eight"},
            {"prompt": "Which gas do plants absorb?",       "completion": "Carbon dioxide"},
            {"prompt": "Translate 'cat' to Spanish.",       "completion": "Gato"},
            {"prompt": "Sun rises in the?",                 "completion": "East"},
        ]

        ds = Dataset.from_list(TOY_ROWS).map(lambda ex: {
            "messages": [
                {"role": "user",      "content": ex["prompt"]},
                {"role": "assistant", "content": ex["completion"]},
            ]
        }, remove_columns=["prompt", "completion"])

        ds = ds.map(_sft_map_fn, remove_columns=["messages"])

        assert all(any(m == 1 for m in r["assistant_mask"]) for r in ds), \
            "some row has zero response tokens — preprocessing or template is off"

        print(f"[ds]  rows                 : {len(ds)}")
        print(f"[ds]  example token lens   : {[len(r['input_ids']) for r in ds]}")
        print(f"[ds]  response tokens / row: {[sum(r['assistant_mask']) for r in ds]}")

        scheduler = LinearAlphaScheduler()
        sampler   = MinimalMDLMSampler(
            backbone=model, scheduler=scheduler, mask_index=tokenizer.mask_token_id,
        )
        sampler.sample_sft = SFTMixin.sample_sft.__get__(sampler, type(sampler))

        collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

        args = MDLMConfig(
            output_dir=tmp,
            num_train_epochs=20,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            logging_steps=1,
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            batch_eval_metrics=True,
            remove_unused_columns=False,
        )

        trainer = MDLMSFTTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            processing_class=tokenizer,
            data_collator=collator,
        )

        trainer.train()