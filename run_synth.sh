#!/bin/bash

if [ ! -d "dial-sft" ]; then
  git clone https://github.com/AvgJoe-cpu/dial-sft.git
fi

cd dial-sft

read -p "Checkpoint path (default: Qwen/Qwen2.5-3B-Instruct): " checkpoint_path
checkpoint_path=${checkpoint_path:-"Qwen/Qwen2.5-3B-Instruct

read -p "Dataset name (default: roskoN/dailydialog): " dataset_name
dataset_name=${dataset_name:-"roskoN/dailydialog"}

read -p "Split (default: validation): " split
split=${split:-"validation"}

read -p "Inference batch size (default: 128): " inference_batch_size
inference_batch_size=${inference_batch_size:-128}

read -p "Output file (default: synthetic_analysis.parquet): " output_file
output_file=${output_file:-"synthetic_analysis.parquet"}
python -c "
from part_c import run_syn
run_syn(
    checkpoint_path='$checkpoint_path',
    dataset_name='$dataset_name',
    split='$split',
    inference_batch_size=$inference_batch_size,
    output_file='$output_file'
)
"