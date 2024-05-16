#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES="1"

# Define variables
arch="vit_t"  # Change this value as needed
finetune_type="lora"
dataset_name="MRI-Prostate"  # Assuming you set this if it's dynamic
targets='combine_all'
# Construct train and validation image list paths
img_folder="./datasets"  # Assuming this is the folder where images are stored
train_img_list="${img_folder}/${dataset_name}/train_5shot.csv"
val_img_list="${img_folder}/${dataset_name}/val_5shot.csv"


# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_decoder_${finetune_type}_${dataset_name}_noprompt"

# Run the Python script
python SingleGPU_train_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_update_encoder True \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -sam_ckpt "mobile_sam.pt" \
    -targets "$targets" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list"