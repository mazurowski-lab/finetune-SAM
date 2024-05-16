#!/bin/bash

arch="vit_h"  # Change this value as needed
finetune_type="lora"
dataset_name="cerebral_edema"  # Assuming you set this if it's dynamic

# Construct the checkpoint directory argument
dir_checkpoint="SAM_${arch}_encoder_${finetune_type}_decoder_${finetune_type}_noprompt_epoch100"

# Run the Python script
python SingleGPU_train_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -img_folder "./datasets" \
    -mask_folder "./datasets" \
    -num_cls 2 \ 
    -sam_ckpt "sam_vit_h_4b8939.pth" \
    -epochs 100 \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"