from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict
import torch


def run_training_cpt(model_path= "EleutherAI/pythia-70m", rev="step512"):

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        revision=rev,
        dtype=torch.bfloat16,
        device_map='auto'
        )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        revision=rev,
        )      
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True,max_length=1024)
    
    # accepts dataset DICT from disk
    dd = DatasetDict.load_from_disk("/content/dummy2.arrow")
    ds_train = dd['train']
    ds_eval  = dd['eval']
    del dd

    #ds_train = ds_train.map(tokenize_function, batched=True, remove_columns=["text"])
    #ds_eval  = ds_eval.map(tokenize_function, batched=True,  remove_columns=["text"])
    ds_train = ds_train.map(tokenize_function, batched=True)
    ds_eval  = ds_eval.map(tokenize_function, batched=True)

    # Standard config
    training_args   = TrainingArguments(
        output_dir  ="./pythia-70m-continued",
        seed=42,

        # --- Core Training ---
        bf16=True,                      
        num_train_epochs=10,
        #max_steps=10,                 
        
        auto_find_batch_size=True,       
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,

        gradient_accumulation_steps=8,
        learning_rate=5.0e-5,
        warmup_steps=10,

        optim="adamw_torch_fused",      
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,     

        # --- Evaluation (FREQUENT for pretraining health) ---
        eval_strategy="steps",     # evaluate every N steps
        eval_steps=100,                 
        eval_delay=0,                    
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,

        # --- Checkpointing ---
        save_strategy="steps",           # align save with eval
        save_steps=100,
        save_total_limit=3,              # keep only last 3 to save disk

        # --- LOGGING 
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        include_num_input_tokens_seen="all",

        log_level="info",              
        log_level_replica="info",    
        logging_nan_inf_filter=False,    # log even NaN/Inf (critical for debug)
        report_to="tensorboard",                   
    )    

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # causal LM (next-token prediction)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset    =ds_train,
        eval_dataset     =ds_eval,     
        processing_class =tokenizer,
        data_collator    =data_collator,
    )

    trainer.train()

    del tokenizer, model, ds_train, ds_eval, trainer, training_args
    torch.cuda.empty_cache()


run_training_cpt()