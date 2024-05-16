# Finetune SAM on your customized medical imaging dataset
Authors: [Hanxue Gu*](https://scholar.google.com/citations?hl=en&user=aGjCpQUAAAAJ&view_op=list_works&sortby=pubdate), [Haoyu Dong*](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en), [Jichen Yang](https://scholar.google.com/citations?user=jGv3bRUAAAAJ&hl=en), [Maciej A. Mazurowski](https://scholar.google.com/citations?user=HlxjJPQAAAAJ&hl=en)

This is the official code for our paper: [How to build the best medical image segmentation algorithm using foundation models: a comprehensive empirical study with Segment Anything Model](https://arxiv.org/abs/2404.09957), where we explore three popular scenarios when fine-tuning foundation models to customized datasets in the medical imaging field: (1) only a single labeled dataset; (2) multiple labeled datasets for different tasks; and (3) multiple labeled and unlabeled datasets; and we design three common experimental setups, as shown in figure 1.
![Fig1: Overview of general fine-tuning strategies based on different levels of dataset availability.](https://github.com/mazurowski-lab/finetune-SAM/blob/main/finetune_strategy_v9.png)

Our work summarizes and evaluates existing fine-tuning strategies with various backbone architectures,  model components, and fine-tuning algorithms across 18 combinations, and 17 datasets covering all common radiology modalities. 
![Fig2: Visualization of task-specific fine-tuning architectures selected in our study: including 3 encoder architecture $\times$ 2 model components $\times$ 3 vanilla/PEFT methods = 18 choices.](https://github.com/mazurowski-lab/finetune-SAM/blob/main/finetune_combination_v3.png)


Based on our extensive experiments, we found that:
1.  fine-tuning SAM leads to slightly better performance than previous segmentation methods.
2. fine-tuning strategies that use parameter-efficient learning in both the encoder and decoder are superior to other strategies.
3. network architecture has a small impact on the final performance, 
4. further training SAM with self-supervised learning can improve final model performance.

To use our codebase, we provide (a) codes to fine-tune your medical imaging dataset on either automatic/prompt-based setting, (b) pretrained weights we got from Setup 3 using task-agnostic self-supervised learning, which we found as good pretrained weights instead of initial SAM providing better performance for downstream tasks.

## Bug fixes:
- [X] May-10-2024, fixed the bug that when we updated the dataset.py at May 6th for multi class support, the mask resize processing was accidently forgotten.
- [X] May-10-2024, fixed the bug that the provided demo for single gpu trianing only support updating decoder but the image encoder's gradients were not calculated.

## Updated functions:
- [X] May-15-2024, add functions to auto save training args and load args for validation; save your time for manual definition.
- [X] May-15-2024, add two jupyter-notebooks showing examples about how to make predictions on 3D volumes/2D pngs without ground truth; and for visualization.
- [X] May-15-2024, provide two additional example demos. 


## a): fine-tune to one single task-specific dataset 
### Step 0: setup environment
```bash
conda env create -f environment.yml
```

### Step 1: dataset preparation.
Please prepare your images and mask pairs in 2D slices first. If your original dataset is in 3D format, please preprocess it and save images/masks as 2D slices.

There is no strict format for your dataset folder; you need first to identify your main dataset folder, for example:
```
args.img_folder = './datasets/'
args.mask_folder = './datasets/'
```
Then prepare your image/mask list file train/val/test.csv under **args.img_folder/dataset_name/** in the following format: ``img_slice_path mask_slice_path``, such as:
```
sa_xrayhip/images/image_044.ni_z001.png	sa_xrayhip/masks/image_044.ni_z001.png
sa_xrayhip/images/image_126.ni_z001.png	sa_xrayhip/masks/image_126.ni_z001.png
sa_xrayhip/images/image_034.ni_z001.png	sa_xrayhip/masks/image_034.ni_z001.png
sa_xrayhip/images/image_028.ni_z001.png	sa_xrayhip/masks/image_028.ni_z001.png
```
## Step 2:
Configure your network architectures and other hyperparameters.
### (1) Choose image encoder architecture.
```
args.arch = 'vit_b' # you can pick from  'vit_h','vit_b','vit_t'

#If load original sam's encoder, for example, if 'vit_b':
args.sam_ckpt = "sam_vit_b_01ec64.pth" 
# You can replace it with any other pretrained weights, such as 'medsam_vit_b.pth'
```
You need to download SAM's checkpoints of vit-h, and vit-b from [SAM](https://github.com/facebookresearch/segment-anything),  and to use MobileSAM; you can download the checkpoints from [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

**To be noticed****
If pretrained weights are used as MedSAM, you need to use dataset normalization as [0-1] instead of the original SAM's mean/std normations.
```
# normalzie_type: 'sam' or 'medsam', if sam, using transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); if medsam, using [0,1] normalize.

args.normalize_type = 'medsam'
```

### (2) Choose fine-tuning Methods.

#### (i) Vanilla fine-tuning
 - If you want to update Encoder and Decoder both, just load the network and put:
```
args.if_update_encoder = True
```
 - If you only want to update Mask Decoder, just load the network and put:
```
args.if_update_encoder = False
```

#### (2) fine-tuning using Adapter blocks
- If you want to add adapter blocks on the image encoder and mask decoder both:
```
args.if_mask_decoder_adapter=True

args.if_update_encoder = True
args.if_encoder_adapter=True
# You can pick the image encoder blocks by adding adapters
args.encoder_adapter_depths = range(0,12)
```
- If you want to add adapter blocks to the decoder only:
```
args.if_mask_decoder_adapter=True
```

#### (3) fine-tuning using LoRA blocks
-  If you want to add LoRA blocks on the image encoder and mask decoder both:
```
# define which blocks you would like to add LoRAs, if [] is empty, it will be added at **each** block.
args.if_update_encoder = True
args.if_encoder_lora_layer = True
args.encoder_lora_layer = []
args.if_decoder_lora_layer = True  
```
- If you only want to add LoRA blocks on the mask decoder:
```
args.if_decoder_lora_layer = True  
```

### Other configurations
1. If you want to enable warmup:
```
# If you want to use warmup
args.if_warmup = True
args.warmup_period = 200
```
2. If you want to use DDP training for multiple GPUs, use 
```
python DDP_train_xxx.py
```
Otherwise, use:
```
python SingleGPU_train_xxx.py
```
if the network is large and you cannot fit into one single GPU, you can use our DDP_train_xxx.py as well as split the image encoder into 2 GPUs:
```
args.if_split_encoder_gpus = True
args.gpu_fractions = [0.5,0.5] # the fraction of image encoder on each GPU
```

### Multi-cls segmentation VS. binary segmentation
1. if you want to do binary segmentation:
```
# set the output channels as 2 (background, object)
args.num_cls = 2
```

If your target objects actually have multiple labels but you want to combine them as binary:
```
# put the dataset's parameter for 'target' as 'combine_all', for example:
Public_dataset(args,args.img_folder, args.mask_folder, train_img_list,phase='train',targets=['combine_all'],normalize_type='sam',if_prompt=False)
```
2. if you want to do multi-cls segmentation:
```
# set the output channels as num_of_target_objects + 1 (background, object1, object2,...)
args.num_cls = n+1

# put the dataset's parameter for 'target' as 'multi_all', for example:
Public_dataset(args,args.img_folder, args.mask_folder, train_img_list,phase='train',targets=['multi_all'],normalize_type='sam',if_prompt=False)
```

3. if you actually have multiple different targets but you want to select a subset, such as one target from your mask for trianing:
```
Todo
```

### Example bash file for running the training
Here is one example (train_singlegpu_demo.sh) of running the training on a demo dataset using **vit-b** with **Adapter** and updating **Mask Decoder** only.
```
#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES="5"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="adapter"
dataset_name="MRI-Prostate"  # Assuming you set this if it's dynamic
targets='combine_all' # make it as binary segmentation 'multi_all' for multi cls segmentation
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
    -if_mask_decoder_adapter True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -sam_ckpt "sam_vit_b_01ec64.pth" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list"
```
To run the training, just use the command:
```
bash train_singlegpu_demo.sh
or 
bash train_ddpgpu_demo.sh
```

### Visualization of the loss
You can visualize your training logs using tensorboard; in a terminal, just type:
```
tensorboard --logdir args.dir_checkpoint/log --ip 0.0.0.0
```
Then, open the browser to visualize the loss.


### Additional interactive modes
if you want to use prompt_based training, just edit the dataset into **prompt_type='point' or prompt_type='box' or prompt_type='hybrid'**, for example:
```
train_dataset = Public_dataset(args,args.img_folder, args.mask_folder, train_img_list,phase='train',targets=['all'],normalize_type='sam',prompt_type='point')
eval_dataset = Public_dataset(args,args.img_folder, args.mask_folder, val_img_list,phase='val',targets=['all'],normalize_type='sam',prompt_type='point')
```
And you need to edit the block for the prompt encoder input accordingly:
```
sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
```
## Step 3: Validation of the model
```
bash val_singlegpu_demo.sh
```

## Additional model inference mode and prediction visualization
Refer to  2D_predictions_with_vis.ipynb and 3D_predictions_with_vis.ipynb.


## b): fine-tune from task-expansive pretrained weights
If you want to use MedSAM as pretrained weights, please refer to [MedSAM](https://github.com/bowang-lab/MedSAM) and download their checkpoints as 'medsam_vit_b.pth'.

## c): fine-tune from task-agnostic self-supervised pre-trained weights
In our paper, we found that training in Setup 3, which starts from self-supervised weights and then fine-tuning to one customized dataset using Parameter Efficient Learning to fine-tune both Encoder/Decoder, provides the best model.
To use our self-supervised pretrained weights, please refer to [SSLSAM](https://drive.google.com/drive/folders/1JAoy-Mh5QgxXsjWtQhMjOX16dN1kytLQ).

## ToDOlist:
 - [x] add the branch of codes for automatic multi-cls segmentation
 - [ ] add the branch of codes for prompt-based multi-cls segmentation. output has two channels and random select one target at one time during training.


## Acknowledgement
This work was supported by Duke Univeristy.
We built these codes based on the following:
1. [SAM](https://github.com/facebookresearch/segment-anything)
2. [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
3. [MedSAM](https://github.com/bowang-lab/MedSAM)
4. [Medical SAM Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)
5. [LoRA for SAM](https://github.com/JamesQFreeman/Sam_LoRA)

## Citation
Please cite our paper if you use our code or reference our work:
```bib
@misc{gu2024build,
      title={How to build the best medical image segmentation algorithm using foundation models: a comprehensive empirical study with Segment Anything Model}, 
      author={Hanxue Gu and Haoyu Dong and Jichen Yang and Maciej A. Mazurowski},
      year={2024},
      eprint={2404.09957},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
