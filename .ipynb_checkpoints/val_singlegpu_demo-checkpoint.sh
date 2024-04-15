#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES="5"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="adapter"
dataset_name="MRI-Prostate"  # Assuming you set this if it's dynamic

# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_decoder_${finetune_type}_${dataset_name}_noprompt"

# Run the Python script
python val_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_mask_decoder_adapter True \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"