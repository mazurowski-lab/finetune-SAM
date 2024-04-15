#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES="5,6"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type= "vanilla"
dataset_name="MRI-Prostate"  # Assuming you set this if it's dynamic

# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_encoderdecoder_${finetune_type}_${dataset_name}_noprompt"

# Run the Python script
python DDP_splitgpu_train_finetune_noprompt.py \
    -if_warmup True \
    -if_split_encoder_gpus True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -sam_ckpt "sam_vit_b_01ec64.pth" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"