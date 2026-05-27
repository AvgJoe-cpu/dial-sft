# (https://huggingface.co/docs/trl/sft_trainer$0)
# standard - conversational
# LM - prompt completion

from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from jinja2 import Template

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


#dd = load_dataset("euclaise/writingprompts")
#print(dd)
#train_ds = dd['train'].select(range(1000)).rename_column('story', 'completion')
#eval_ds  = dd['validation'].select(range(100)).rename_column('story', 'completion')
#print(train_ds)

#test_ds  = dd['test'].select(range(10)).rename_column('story', 'completion')

#def count_tokens_in_column(example, tokenizer=None, column_name=None):
#    tokenized = tokenizer(example[column_name])
#    token_count = len(tokenized["input_ids"])
#    count_column_name = f"{column_name}_token_count"
#    return {count_column_name: token_count}
#
#dd = load_dataset("euclaise/writingprompts")
#train_ds = dd['train']
#eval_ds  = dd['validation']
#test_ds  = dd['test']#
#
#_, tokenizer = setup_model_and_tokenizer(model_name="EleutherAI/pythia-160m")#
#
#train_ds = train_ds.map(
#    count_tokens_in_column_batched,
#    batched=True,
#    fn_kwargs={
#        "tokenizer": tokenizer,
#        "column_name": "prompt",
#    },
#).map(
#    count_tokens_in_column_batched,
#    batched=True,
#    fn_kwargs={
#        "tokenizer": tokenizer,
#        "column_name": "story",
#    },
#)
#
#eval_ds = eval_ds.map(
#    count_tokens_in_column_batched,
#    batched=True,
#    fn_kwargs={
#        "tokenizer": tokenizer,
#        "column_name": "prompt",
#    },
#).map(
#    count_tokens_in_column_batched,
#    batched=True,
#    fn_kwargs={
#        "tokenizer": tokenizer,
#        "column_name": "story",
#    },
#)
#
#test_ds = test_ds.map(
#    count_tokens_in_column_batched,
#    batched=True,
#    fn_kwargs={
#        "tokenizer": tokenizer,
#        "column_name": "prompt",
#    },
#).map(
#    count_tokens_in_column_batched,
#    batched=True,
#    fn_kwargs={
#        "tokenizer": tokenizer,
#        "column_name": "story",
#    },
#)
#
#train_ds.save_to_disk(TRAIN_SAVE_PATH)
#eval_ds.save_to_disk(EVAL_SAVE_PATH)
#test_ds.save_to_disk(TEST_SAVE_PATH)

def count_tokens_in_column_batched(batch, tokenizer=None, column_name: str = "story"):
    tokenized = tokenizer(batch[column_name], truncation=False, padding=False)
    token_counts = [len(input_ids) for input_ids in tokenized["input_ids"]]
    count_column_name = f"{column_name}_token_count"
    return {count_column_name: token_counts}

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
    train_ds = dd['train']
    eval_ds = dd['validation']
    test_ds = dd['test']
    del dd
    gc.collect()

    print("Processing train split...")
    train_ds = train_ds.map(
        count_tokens_in_column_batched,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "column_name": "prompt",
        },
    ).map(
        count_tokens_in_column_batched,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "column_name": "story",
        },
    )
    print("Processing eval split...")
    eval_ds = eval_ds.map(
        count_tokens_in_column_batched,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "column_name": "prompt",
        },
    ).map(
        count_tokens_in_column_batched,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "column_name": "story",
        },
    )
    print("Processing test split...")
    test_ds = test_ds.map(
        count_tokens_in_column_batched,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "column_name": "prompt",
        },
    ).map(
        count_tokens_in_column_batched,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "column_name": "story",
        },
    )
    print(f"Saving train split to: {train_save_path}")
    train_ds.save_to_disk(train_save_path)
    del train_ds
    gc.collect()

    print(f"Saving eval split to: {eval_save_path}")
    eval_ds.save_to_disk(eval_save_path)
    del eval_ds
    gc.collect()

    print(f"Saving test split to: {test_save_path}")
    test_ds.save_to_disk(test_save_path)
    del test_ds
    gc.collect()

    print("✓ All splits processed and saved!")

#_, tokenizer = setup_model_and_tokenizer(model_name="EleutherAI/pythia-160m")
#process_and_save_datasets(dataset_name="euclaise/writingprompts", tokenizer=tokenizer, train_save_path="./train_ds_og", eval_save_path="./eval_ds_og", test_save_path="./test_ds_og")
#del tokenizer

def setup_model_and_tokenizer(
    model_name: str,
    for_training: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):

    chat_template_str = """
    {%- for message in messages %}
    {%- if message['role'] == 'system' %}
    {{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' %}
    {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
    {{- '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{- '<|im_end|>' }}
    {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- endif %}
    """.strip()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = chat_template_str

    special_tokens_dict = {
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right" if for_training else "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map
    )
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=64
    )
    return model, tokenizer

#setup_model_and_tokenizer(model_name="EleutherAI/pythia-160m")

#user_templ = Template("""
#Continue the story based on the beginning:
#{{ Prompt }}
#""")
#
#asst_templ = Template("""
#Based on the begnning, the story could continue like this:
#{{ Story }}
#""")

#def add_prompt_fn(example):
#    prompt = user_templ.render(Prompt=example["prompt"])
#    completion = asst_templ.render(Story=example["story"])
#
#    example["prompt"] = prompt
#    example["completion"] = prompt
#    return example

#dd = load_dataset("euclaise/writingprompts")
#ds = dd['test']

#ds = ds.map(
#    add_prompt_fn,
#
#)
#print(ds["prompt"][0])

#chat_template_str = """
#{%- for message in messages %}
#{%- if message['role'] == 'system' %}
#{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
#{%- elif message['role'] == 'user' %}
#{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
#{%- elif message['role'] == 'assistant' %}
#{{- '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{- '<|im_end|>' }}
#{%- endif %}
#{%- endfor %}
#{%- if add_generation_prompt %}
#{{- '<|im_start|>assistant\n' }}
#{%- endif %}
#""".strip()

#MODEL_NAME = "EleutherAI/pythia-160m"
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#tokenizer.chat_template = chat_template_str
#special_tokens_dict = {
#    "additional_special_tokens": [
#        "<|im_start|>",
#        "<|im_end|>",
#        "<|user|>",
#        "<|assistant|>",
#        "<|system|>"
#    ]
#}
#tokenizer.add_special_tokens(special_tokens_dict)
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token

#model = AutoModelForCausalLM.from_pretrained(
#    MODEL_NAME,
#    dtype=torch.bfloat16,
#    device_map='auto'
#)

#model.resize_token_embeddings(
#  len(tokenizer),
#  pad_to_multiple_of=64
#)
#model, tokenizer = setup_model_and_tokenizer(model_name="EleutherAI/pythia-160m")    # NEW
        #output_dir="./sft_output",

def add_prompt_fn(example):

    user_templ = Template("""
    Continue the story based on the beginning:
    {{ Prompt }}
    """)

    asst_templ = Template("""
    Based on the beginning, the story could continue like this:
    {{ Story }}
    """)

    prompt = user_templ.render(Prompt=example["prompt"])
    completion = asst_templ.render(Story=example["completion"])

    example["prompt"] = prompt
    example["completion"] = prompt
    return example


def format_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]}
        ]
    }


#def run_training(MODEL_NAME, OUTPUT_DIR, train_dataset):
#    model, tokenizer = setup_model_and_tokenizer(model_name=MODEL_NAME)
#
#    training_args = SFTConfig(
#        output_dir=OUTPUT_DIR,
#        push_to_hub=False,
#
#        num_train_epochs=2,
#        per_device_train_batch_size=64,
#        logging_steps=10,
#        assistant_only_loss=True,  # Loss ONLY on assistant messages (via {% generation %})
#    )
#
#    trainer = SFTTrainer(
#        model=model,
#        args=training_args,
#        train_dataset=train_dataset,
#        processing_class=tokenizer,  # Tokenizer with chat_template set
#    )
#    trainer.train()
#    trainer.save_model()
#    torch.cuda.empty_cache()
#    del model, tokenizer, training_args, trainer
#
#
#def _run_training(DATASET_PATH, MODEL_NAME, OUTPUT_DIR, is_local: bool = False):
#    if is_local:
#        print(f"Loading dataset from local disk: {DATASET_PATH}")
#        train_ds = Dataset.load_from_disk(DATASET_PATH)
#    else:
#        print(f"Loading dataset from HuggingFace Hub: {DATASET_PATH}")
#        dd = load_dataset(DATASET_PATH)
#        train_ds = dd['train']
#        del dd
#
#    train_ds = train_ds.select(range(10000)).rename_column('story', 'completion')
#
#    train_dataset = train_ds.map(
#        format_to_messages,
#        remove_columns=train_ds.column_names
#    )
#    run_training(MODEL_NAME, OUTPUT_DIR, train_dataset)
#    del MODEL_NAME, train_dataset, train_ds

#DATASET_PATH = "euclaise/writingprompts"
#MODEL_NAME = "EleutherAI/pythia-160m"
#OUTPUT_DIR = "./sft_output"
#_run_training(DATASET_PATH, MODEL_NAME, OUTPUT_DIR)

def run_training(
    TRAIN_DATA_LOAD_PATH: str,
    TRAIN_MODEL_LOAD_PATH: str,
    TRAIN_MODEL_SAVE_PATH: str,
    num_samples: int = 10000,
    num_epochs: int = 4,
    batch_size: int = 64,
    #is_local: bool = False,
):
    output_dir   = TRAIN_MODEL_SAVE_PATH
    model_name   = TRAIN_MODEL_LOAD_PATH
    dataset_path = TRAIN_DATA_LOAD_PATH
    #if is_local:
    #    print(f"Loading dataset from local disk: {dataset_path}")
    #    train_ds = Dataset.load_from_disk(dataset_path)
    #else:
    #    print(f"Loading dataset from HuggingFace Hub: {dataset_path}")
    #    dd = load_dataset(dataset_path)
    #    train_ds = dd['train']
    #    del dd
    train_ds = load_from_disk(dataset_path)

    print(f"Selecting {num_samples} samples and preprocessing...")
    train_ds = train_ds.select(range(num_samples)).rename_column('story', 'completion')
    train_ds = train_ds.map(add_prompt_fn) # NEW

    train_dataset = train_ds.map(
        format_to_messages,
        remove_columns=train_ds.column_names
    )
    del train_ds

    print(f"Loading model: {model_name}")
    model, tokenizer = setup_model_and_tokenizer(model_name=model_name, for_training=True)

    training_args = SFTConfig(
        push_to_hub=False,
        output_dir=output_dir,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/tb_logs",

        bf16=True,
        optim="adamw_torch_fused",
        use_liger_kernel=False, # False on mps 

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=10,

        assistant_only_loss=True,  # Loss ONLY on assistant messages
    )

    print(f"Starting training with {num_samples} samples...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    print(f"✓ Training complete. Model saved to: {output_dir}")
    torch.cuda.empty_cache()
    del model, tokenizer, training_args, trainer, train_dataset

#def tokenize_fn(example):
#    tokenized = tokenizer.apply_chat_template([{"role": "user", "content": example["prompt"]}], tokenize=False, add_generation_prompt=True)
#    return {"text": tokenized}

#def tokenize_fn(batch):
#    conversations = [[{"role": "user", "content": prompt}] for prompt in batch["prompt"]]
#    tokenized = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
#    return {"text": tokenized}


def add_prompt_inference(example):
    user_templ = Template("""
    Continue the story based on the beginning:
    {{ Prompt }}
    """)

    prompt = user_templ.render(Prompt=example["prompt"])
    example["prompt"] = prompt
    return example


#def tokenize_fn(batch, tokenizer=None):
#    messages_list = []
#    for prompt in batch["prompt"]:
#        messages = [{"role": "user", "content": prompt}]
#        messages_list.append(messages)
#
#    formatted_texts = tokenizer.apply_chat_template(
#        messages_list,
#        tokenize=False,
#        add_generation_prompt=True
#    )
#
#    return {"text": formatted_texts}
#
#
#def generate_ar(batch, tokenizer=None, model=None, gen_config=None):
#    formatted_texts = batch["text"]
#
#    model_inputs = tokenizer(
#        formatted_texts,
#        return_tensors="pt",
#        padding=True,
#        truncation=True,
#        max_length=512
#    ).to(model.device)
#
#    generated_ids = model.generate(
#        **model_inputs,
#        generation_config=gen_config,
#    )
#    output_ids = [
#        generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
#        for i in range(len(generated_ids))
#    ]
#    results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#
#    del model_inputs, generated_ids, output_ids
#    return {"story": results}


def generate_ar(batch, tokenizer=None, model=None, gen_config=None):
    messages_list = [[{"role": "user", "content": prompt}] for prompt in batch["prompt"]]
    formatted_texts = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        generation_config=gen_config,
    )

    results = tokenizer.batch_decode(
        [generated_ids[i][len(model_inputs.input_ids[i]):].tolist() for i in range(len(generated_ids))],
        skip_special_tokens=True
    )

    del model_inputs, generated_ids, formatted_texts, messages_list
    return {"story": results}


def run_inference(INFER_MODEL_LOAD_PATH, INFER_DATA_LOAD_PATH, INFER_DATA_SAVE_PATH, num_samples: int=10000):
    MODEL_NAME = INFER_MODEL_LOAD_PATH
    INFERENCE_LOAD_PATH = INFER_DATA_LOAD_PATH
    SAVE_PATH = INFER_DATA_SAVE_PATH

    dataset = load_from_disk(INFERENCE_LOAD_PATH)
    ds = dataset
    del dataset
    ds = ds.select(range(num_samples))

    model, tokenizer = setup_model_and_tokenizer(model_name=MODEL_NAME, for_training=False)    # NEW
    tokenizer.padding_side = "left"

    config = GenerationConfig(
        max_new_tokens=512,
        num_beams=1,
        do_sample=True,
        use_cache=True,
        temperature=1.1,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    ds = ds.map(
        add_prompt_inference,
    )
    #ds = ds.map(
    #    tokenize_fn,
    #    batched=True,
    #    batch_size=32,
    #    fn_kwargs={
    #        "tokenizer": tokenizer,
    #    }
    #)

    ds = ds.map(
        generate_ar,
        batched=True,
        batch_size=10,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "gen_config": config,
        },
    #    remove_columns=["text"]
    )
    ds.save_to_disk(SAVE_PATH)
    torch.cuda.empty_cache()
    del model, tokenizer, config, ds, INFERENCE_LOAD_PATH, SAVE_PATH, MODEL_NAME


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        TRAIN_OG_PATH       = "euclaise/writingprompts"
        TRAIN_OG_SAVE_PATH  = f"{tmp}/train_ds_og"
        EVAL_OG_SAVE_PATH   = f"{tmp}/eval_ds_og"
        TEST_OG_SAVE_PATH   = f"{tmp}/test_ds_og"

        _, tokenizer = setup_model_and_tokenizer(model_name="EleutherAI/pythia-70m")
        process_and_save_datasets(
            dataset_name=TRAIN_OG_PATH,
            tokenizer=tokenizer,
            train_save_path=TRAIN_OG_SAVE_PATH,
            eval_save_path=EVAL_OG_SAVE_PATH,
            test_save_path=TEST_OG_SAVE_PATH,
        )
        del tokenizer

        CONFIG_DICT = {
            "ROUND1": {
                "TRAIN_DATA_LOAD_PATH":  TRAIN_OG_SAVE_PATH,
                "TRAIN_MODEL_LOAD_PATH": "EleutherAI/pythia-70m",
                "TRAIN_MODEL_SAVE_PATH": f"{tmp}/sft_output",
                "INFER_MODEL_LOAD_PATH": f"{tmp}/sft_output",
                "INFER_DATA_LOAD_PATH":  TRAIN_OG_SAVE_PATH,
                "INFER_DATA_SAVE_PATH":  f"{tmp}/local_arrow_dataset",
            },
        }

        for round_name, config in CONFIG_DICT.items():
            print(f"\n{'='*80}")
            print(f"STARTING {round_name}")
            print(f"{'='*80}\n")

            train_data_load_path  = config["TRAIN_DATA_LOAD_PATH"]
            train_model_load_path = config["TRAIN_MODEL_LOAD_PATH"]
            train_model_save_path = config["TRAIN_MODEL_SAVE_PATH"]
            infer_model_load_path = config["INFER_MODEL_LOAD_PATH"]
            infer_data_load_path  = config["INFER_DATA_LOAD_PATH"]
            infer_data_save_path  = config["INFER_DATA_SAVE_PATH"]

            print(f"Training data:        {train_data_load_path}")
            print(f"Training model:       {train_model_load_path}")
            print(f"Will save model to:   {train_model_save_path}")

            # ── TRAINING ─────────────────────────────────────────────────────
            print(f"[{round_name}] Running training...")
            run_training(
                TRAIN_DATA_LOAD_PATH=train_data_load_path,
                TRAIN_MODEL_LOAD_PATH=train_model_load_path,
                TRAIN_MODEL_SAVE_PATH=train_model_save_path,
                num_samples=100,
            )
            print(f"✓ [{round_name}] Training complete\n")

            # ── INFERENCE ────────────────────────────────────────────────────
            print(f"[{round_name}] Running inference...")
            run_inference(
                INFER_MODEL_LOAD_PATH=infer_model_load_path,
                INFER_DATA_LOAD_PATH=infer_data_load_path,
                INFER_DATA_SAVE_PATH=infer_data_save_path,
                num_samples=100,
            )
            print(f"✓ [{round_name}] Inference complete\n")

            print(f"{'='*80}")
            print(f"✓ {round_name} FINISHED")
            print(f"{'='*80}\n")