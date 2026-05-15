#!/bin/bash

# Clone the repository if not already present
if [ ! -d "dial-sft" ]; then
  git clone https://github.com/AvgJoe-cpu/dial-sft.git
fi

cd dial-sft

read -p "Model name (default: facebook/opt-350m): " model_name
model_name=${model_name:-"facebook/opt-350m"}

read -p "Dataset name (default: roskoN/dailydialog): " dataset_name
dataset_name=${dataset_name:-"roskoN/dailydialog"}

read -p "Split (default: train): " split
split=${split:-"train"}

read -p "Output dir (default: ./sft_output): " output_dir
output_dir=${output_dir:-"./sft_output"}

# Execute the training function with the provided arguments
python -c "
from part_a import run_training
run_training(
    model_name='$model_name',
    dataset_name='$dataset_name',
    split='$split',
    output_dir='$output_dir'
)
"