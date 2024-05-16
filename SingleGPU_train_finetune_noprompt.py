#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from skimage.measure import label
from models.sam_LoRa import LoRA_Sam
#Scientific computing 
import numpy as np
import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from tensorboardX import SummaryWriter
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from utils.losses import DiceLoss
from utils.dsc import dice_coeff_multi_class
import cv2
import monai
from utils.utils import vis_image
import cfg
import json
# Use the arguments
args = cfg.parse_args()
# you need to modify based on the layer of adapters you are choosing to add
# comment it if you are not using adapter
#args.encoder_adapter_depths = [0,1,2,3]

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def train_model(trainloader,valloader,dir_checkpoint,epochs):
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
    
    sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
    if args.finetune_type == 'adapter':
        for n, value in sam.named_parameters():
            if "Adapter" not in n: # only update parameters in adapter
                value.requires_grad = False
        print('if update encoder:',args.if_update_encoder)
        print('if image encoder adapter:',args.if_encoder_adapter)
        print('if mask decoder adapter:',args.if_mask_decoder_adapter)
        if args.if_encoder_adapter:
            print('added adapter layers:',args.encoder_adapter_depths)
        
    elif args.finetune_type == 'vanilla' and args.if_update_encoder==False:   
        print('if update encoder:',args.if_update_encoder)
        for n, value in sam.image_encoder.named_parameters():
            value.requires_grad = False
    elif args.finetune_type == 'lora':
        print('if update encoder:',args.if_update_encoder)
        print('if image encoder lora:',args.if_encoder_lora_layer)
        print('if mask decoder lora:',args.if_decoder_lora_layer)
        sam = LoRA_Sam(args,sam,r=4).sam
    sam.to('cuda')
        
    optimizer = optim.AdamW(sam.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True,reduction='mean')
    criterion2 = nn.CrossEntropyLoss()
    
    iter_num = 0
    max_iterations = epochs * len(trainloader) 
    writer = SummaryWriter(dir_checkpoint + '/log')
    
    pbar = tqdm(range(epochs))
    val_largest_dsc = 0
    last_update_epoch = 0
    for epoch in pbar:
        sam.train()
        train_loss = 0
        for i,data in enumerate(tqdm(trainloader)):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
            msks = msks.cuda()

            if args.if_update_encoder:
                img_emb = sam.image_encoder(imgs)
            else:
                with torch.no_grad():
                    img_emb = sam.image_encoder(imgs)
            
            # get default embeddings
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, _ = sam.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
            loss_dice = criterion1(pred,msks.float()) 
            loss_ce = criterion2(pred,torch.squeeze(msks.long(),1))
            loss =  loss_dice + loss_ce
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            else:
                if args.if_warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

            train_loss += loss.item()
            iter_num+=1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

        train_loss /= (i+1)
        pbar.set_description('Epoch num {}| train loss {} \n'.format(epoch,train_loss))

        if epoch%2==0:
            eval_loss=0
            dsc = 0
            sam.eval()
            with torch.no_grad():
                for i,data in enumerate(tqdm(valloader)):
                    imgs = data['image'].cuda()
                    msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
                    msks = msks.cuda()

                    img_emb= sam.image_encoder(imgs)
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = sam.mask_decoder(
                                    image_embeddings=img_emb,
                                    image_pe=sam.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=sparse_emb,
                                    dense_prompt_embeddings=dense_emb, 
                                    multimask_output=True,
                                  )
                    loss = criterion1(pred,msks.float()) + criterion2(pred,torch.squeeze(msks.long(),1))
                    eval_loss +=loss.item()
                    dsc_batch = dice_coeff_multi_class(pred.argmax(dim=1).cpu(), torch.squeeze(msks.long(),1).cpu().long(),args.num_cls)
                    dsc+=dsc_batch

                eval_loss /= (i+1)
                dsc /= (i+1)
                
                writer.add_scalar('eval/loss', eval_loss, epoch)
                writer.add_scalar('eval/dice', dsc, epoch)
                
                print('Eval Epoch num {} | val loss {} | dsc {} \n'.format(epoch,eval_loss,dsc))
                if dsc>val_largest_dsc:
                    val_largest_dsc = dsc
                    last_update_epoch = epoch
                    print('largest DSC now: {}'.format(dsc))
                    torch.save(sam.state_dict(),dir_checkpoint + '/checkpoint_best.pth')
                elif (epoch-last_update_epoch)>20:
                    # the network haven't been updated for 20 epochs
                    print('Training finished###########')
                    break
    writer.close()
                
                
                
if __name__ == "__main__":
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name)) 
    train_img_list = args.train_img_list
    val_img_list = args.val_img_list
    
    num_workers = 8
    if_vis = True
    Path(args.dir_checkpoint).mkdir(parents=True,exist_ok = True)
    path_to_json = os.path.join(args.dir_checkpoint, "args.json")
    args_dict = vars(args)
    with open(path_to_json, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    print(args.targets)

    train_dataset = Public_dataset(args,args.img_folder, args.mask_folder, train_img_list,phase='train',targets=[args.targets],normalize_type='sam',if_prompt=False)
    eval_dataset = Public_dataset(args,args.img_folder, args.mask_folder, val_img_list,phase='val',targets=[args.targets],normalize_type='sam',if_prompt=False)
    trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers)

    train_model(trainloader,valloader,args.dir_checkpoint,args.epochs)