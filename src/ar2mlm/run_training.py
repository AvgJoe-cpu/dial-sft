from src.ar2mlm.mlm_scheduler import CosineAlphaScheduler
from src.ar2mlm.mlm_trainer import MDLMTrainer

from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
import torch 

# def pre_process_pt(dataset):
#def do_something_fn(example, tokenizer=None, input_key="text", output_key="input_ids", return_attention_mask=False):
#    return {output_key: tokenizer(example[input_key], add_special_tokens=False, truncation=False, return_attention_mask=return_attention_mask, return_tensors="pt")}

# def pre_process_sft(dataset):


# from src.ar2mlm.mlm_trainer import MDLMTrainer
def run_mlm_pt(dataset, model_name="a2d-gpt-neox", output_dir="./mlm_output", training_args=None):
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if training_args is None:  # Use defaults if not provided
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            save_steps=10_000,
            save_total_limit=2,
        )

    trainer = MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        scheduler=CosineAlphaScheduler(),
        time_epsilon=1e-3,
        data_collator=DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
            ),        
    )
    trainer.train()

    del trainer, model
    torch.cuda.empty_cache()

    
# (https://github.com/ZHZisZZ/dllm/blob/main/examples/a2d/mdlm/sft.py)
#def run_mlm_sft():
