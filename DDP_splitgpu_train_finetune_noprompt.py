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
from utils.dsc import dice_coeff
import cv2
import monai
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import vis_image
import cfg
args = cfg.parse_args()

def cleanup():
    dist.destroy_process_group()
    
def setup(rank, world_size, model_basic, trainloader, valloader,dir_checkpoint, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12333'
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    model_basic(args,rank, world_size,trainloader,valloader,dir_checkpoint)

                    
def model_basic(args,rank, world_size,trainloader,valloader,dir_checkpoint):
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    
    args.devices = [dev0,dev1]
    
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
    

    epochs = args.epochs
    iter_num = 0
    max_iterations = epochs * len(trainloader) 
    writer = SummaryWriter(dir_checkpoint + '/log')
    
    print(f"Running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    model = sam_model_registry["vit_b"](args,checkpoint=os.path.join("sam_vit_b_01ec64.pth"),num_classes=2)
    #print(model)

    if args.finetune_type == 'adapter':
        for n, value in model.named_parameters():
            if "Adapter" not in n: # only update parameters in adapter
                value.requires_grad = False
    elif args.finetune_type == 'vanilla' and args.if_update_encoder==False:      
        for n, value in model.image_encoder.named_parameters():
            value.requires_grad = False
    elif args.finetune_type == 'lora':
        model = LoRA_Sam(args,model,r=4).sam
        
    
    ddp_model = DDP(model)
    
    optimizer = optim.AdamW(ddp_model.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    optimizer.zero_grad()
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True,reduction='mean')
    criterion2 = nn.CrossEntropyLoss()
    pbar = tqdm(range(epochs))
    val_largest_dsc = 0
    last_update_epoch = 0
    for epoch in pbar:
        ddp_model.train()
        train_loss = 0
        for i,data in enumerate(trainloader):
            imgs = data['image'].to(dev0)
            msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
            msks = msks.to(dev1) # output will be in device 1
            img_emb= ddp_model.module.image_encoder(imgs)
            sparse_emb, dense_emb = ddp_model.module.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            )
            pred, _ = ddp_model.module.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=ddp_model.module.prompt_encoder.get_dense_pe(), 
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
            ddp_model.eval()
            with torch.no_grad():
                for i,data in enumerate(valloader):
                    imgs = data['image'].to(dev0)
                    msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
                    msks = msks.to(dev1)
                    img_emb= ddp_model.module.image_encoder(imgs)
                    sparse_emb, dense_emb = ddp_model.module.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    )
                    pred, _ = ddp_model.module.mask_decoder(
                                    image_embeddings=img_emb,
                                    image_pe=ddp_model.module.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=sparse_emb,
                                    dense_prompt_embeddings=dense_emb, 
                                    multimask_output=True,
                                  )
            
                    loss = criterion1(pred,msks.float()) + criterion2(pred,torch.squeeze(msks.long(),1))
                    eval_loss +=loss.item()
                    dsc_batch = dice_coeff((pred[:,1,:,:].cpu()>0).long(),msks.cpu().long()).item()
                    dsc+=dsc_batch

                eval_loss /= (i+1)
                dsc /= (i+1)
                writer.add_scalar('eval/loss', eval_loss, epoch)
                writer.add_scalar('eval/dice', dsc, epoch)
                
                print('***Eval Epoch num {} | val loss {} | dsc {} \n'.format(epoch,eval_loss,dsc))
                if dsc>val_largest_dsc:
                    val_largest_dsc = dsc
                    last_update_epoch = epoch
                    print('largest DSC now: {}'.format(dsc))
                    Path(dir_checkpoint).mkdir(parents=True,exist_ok = True)
                    torch.save(ddp_model.module.state_dict(),dir_checkpoint + '/checkpoint_best.pth')
                elif (epoch-last_update_epoch)>20:
                    print('Training finished####################')
                    # the network haven't been updated for 20 epochs
                    break
                    
    writer.close()   
    cleanup()
    

def run_demo(demo_fn, size, model_basic,trainloader,valloader,dir_checkpoint):
    mp.spawn(demo_fn,
             args=(size, model_basic,trainloader,valloader,dir_checkpoint),
             nprocs=size,
             join=True)
    
if __name__ == "__main__":
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name)) 
    train_img_list = args.img_folder + dataset_name + '/train_5shot.csv'
    val_img_list = args.img_folder + dataset_name + '/val_5shot.csv'

    num_workers = 0
    if_vis = True

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    size = n_gpus//2



    train_dataset = Public_dataset(args,args.img_folder, args.mask_folder, train_img_list,phase='train',targets=['all'],normalize_type='sam',if_prompt=False)
    eval_dataset = Public_dataset(args,args.img_folder, args.mask_folder, val_img_list,phase='val',targets=['all'],normalize_type='sam',if_prompt=False)
    trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers)

    #processes = []
    #mp.set_start_method('spawn')


    run_demo(setup, size, model_basic,trainloader,valloader,args.dir_checkpoint)