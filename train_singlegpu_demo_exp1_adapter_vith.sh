#!/bin/bash -l
#SBATCH -C ampere
#SBATCH --job-name=stuart_SAM_mask_decoder_all_encoder
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=8-00:00:00
#SBATCH -p gpu,overflow
#SBATCH -G 1
#SBATCH --output=lora_vith.out
#SBATCH --error=lora_vith.err
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=min.huang@emory.edu     # Where to send mail
## Activate the custom python environment
source activate Duke_Sam

## bash train_singlegpu_demo_exp2.sh

# Define variables
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
    -if_encoder_lora_layer False \
    -if_decoder_lora_layer True \
    -img_folder="/labs/kamaleswaranlab/monailabel/Convert_photos/"\
    -mask_folder="/labs/kamaleswaranlab/monailabel/Convert_photos/"\
    -sam_ckpt "/labs/collab/Omics/mihir_RNA/sam_model/sam_vit_h_4b8939.pth" \
    -epochs=100\
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -num_cls 2