#!/bin/bash

cd dial-sft

read -p "Checkpoint path (default: ./sft_output/checkpoint-500): " checkpoint_path
checkpoint_path=${checkpoint_path:-"./sft_output/checkpoint-500"}

read -p "Dataset name (default: roskoN/dailydialog): " dataset_name
dataset_name=${dataset_name:-"roskoN/dailydialog"}

read -p "Split (default: train): " split
split=${split:-"train"}

read -p "Inference batch size (default: 128): " inference_batch_size
inference_batch_size=${inference_batch_size:-128}

read -p "Output file (default: ./inference_output.parquet): " output_file
output_file=${output_file:-"./inference_output.parquet"}

python -c "
from part_b import run_inferene
run_inferene(
    checkpoint_path='$checkpoint_path',
    dataset_name='$dataset_name',
    split='$split',
    inference_batch_size=$inference_batch_size,
    output_file='$output_file'
)
