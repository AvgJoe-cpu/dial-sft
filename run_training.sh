#!/bin/bash

# Install whiptail if needed (Colab)
apt-get install -q -y whiptail 2>/dev/null

# Clone the repository if not already present
if [ ! -d "dial-sft" ]; then
  git clone https://github.com/AvgJoe-cpu/dial-sft.git
fi

cd dial-sft

model_name=$(whiptail --inputbox "Model name:" 8 60 "facebook/opt-350m" --title "Training Config" 3>&1 1>&2 2>&3)
dataset_name=$(whiptail --inputbox "Dataset name:" 8 60 "roskoN/dailydialog" --title "Training Config" 3>&1 1>&2 2>&3)
split=$(whiptail --inputbox "Split:" 8 60 "train" --title "Training Config" 3>&1 1>&2 2>&3)
output_dir=$(whiptail --inputbox "Output dir:" 8 60 "./sft_output" --title "Training Config" 3>&1 1>&2 2>&3)

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