#!/bin/bash

if [ ! -d "dial-sft" ]; then
  git clone https://github.com/AvgJoe-cpu/dial-sft.git
fi
cd dial-sft

echo
echo "  configure run"
echo "  ─────────────"

ask() {
  local prompt="$1" default="$2" var="$3"
  read -p "  $prompt (default: $default): " input
  printf -v "$var" '%s' "${input:-$default}"
}

ask "model name"  "facebook/opt-350m"    model_name
ask "dataset"     "roskoN/dailydialog"   dataset_name
ask "split"       "train"                split
ask "output dir"  "./sft_output"         output_dir

echo
echo "  ─────────────"
echo "  model    →  $model_name"
echo "  dataset  →  $dataset_name"
echo "  split    →  $split"
echo "  output   →  $output_dir"
echo "  ─────────────"
echo
# Execute the training function, now passing the SFT config explicitly
python -c "
from trl import SFTConfig
from src.sft import run_training

# Create the SFT config with defaults (matching your original function)
training_args = SFTConfig(
    output_dir='$output_dir',
    max_steps=500,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    auto_find_batch_size=True,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    bf16=True,
    completion_only_loss=False,
    optim='adamw_torch_fused',
    learning_rate=1e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    seed=42,
)

# Pass the config to run_training
run_training(
    model_name='$model_name',
    dataset_name='$dataset_name',
    split='$split',
    output_dir='$output_dir',
    training_args=training_args
)
"