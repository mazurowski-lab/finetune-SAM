import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import random
import torchio as tio
import slicerio
import nrrd
import monai
import pickle
import nibabel as nib
from scipy.ndimage import zoom
import einops
from utils.funcs import *
from torchvision.transforms import InterpolationMode
#from .utils.transforms import ResizeLongestSide

class Public_dataset(Dataset):
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',sample_num=50,channel_num=1,normalize_type='sam',crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=1,if_prompt=True,prompt_type='point',region_type='largest_3'):
        '''
        normalzie_type: 'sam' or 'medsam', if sam, using transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); if medsam, using [0,1] normalize
        cls: the target cls for segmentation
        prompt_type: point or box
        
        '''
        super(Public_dataset, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.segment_names_to_labels = []
        self.args = args
        self.cls = cls
        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        self.normalize_type = normalize_type
        
        for i,tag in enumerate(targets):
            self.segment_names_to_labels.append((tag,i))
            
        if phase == 'train':
            namefiles = open(img_list,'r')
            self.data_list = namefiles.read().split('\n')[:-1]
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                mask_path = data.split(',')[1]
                if mask_path.startswith('/'):
                    mask_path = mask_path[1:]
                msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
                mask_cls = np.array(msk,dtype=int)
                if part_list[0]=='all' and np.sum(mask_cls)>0:
                    keep_idx.append(idx) 
                elif np.sum(mask_cls)>0:
                    if_keep = False
                    for part in part_list:
                        if mask_path.find(part)>=0:
                            if_keep = True
                    if if_keep:
                        keep_idx.append(idx) 
            print('num with non-empty masks',len(keep_idx),'num with all masks',len(self.data_list))        
            self.data_list = [self.data_list[i] for i in keep_idx] # keep the slices that contains target mask
  
        elif phase == 'val':
            namefiles = open(img_list,'r')
            self.data_list = namefiles.read().split('\n')[:-1]
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                mask_path = data.split(',')[1]
                if mask_path.startswith('/'):
                    mask_path = mask_path[1:]
                msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
                #mask_cls = np.array(np.array(msk,dtype=int)==self.cls,dtype=int)
                mask_cls = np.array(msk,dtype=int)
                if part_list[0]=='all' and np.sum(mask_cls)>0:
                    keep_idx.append(idx) 
                elif np.sum(mask_cls)>0:
                    if_keep = False
                    for part in part_list:
                        if mask_path.find(part)>=0:
                            if_keep = True
                    if if_keep:
                        keep_idx.append(idx) 
            print('num with non-empty masks',len(keep_idx),'num with all masks',len(self.data_list))
            self.data_list = [self.data_list[i] for i in keep_idx]

        if phase == 'train':
            transform_img = [transforms.RandomEqualize(p=0.1),
                 transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                 transforms.ToTensor(),   
                 ]
        else:
            transform_img = [transforms.ToTensor(),
                             ]
        if self.normalize_type=='sam':
            transform_img.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform_img = transforms.Compose(transform_img)
            
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(',')[0]
        mask_path = data.split(',')[1]
        #print(img_path,mask_path)
        img = Image.open(os.path.join(self.img_folder,img_path)).convert('RGB')
        if mask_path.startswith('/'):
            mask_path = mask_path[1:]
        msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
        
        img = transforms.Resize((self.args.image_size,self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        state = torch.get_rng_state()
        if self.crop:
            im_w, im_h = img.size
            diff_w = max(0,self.crop_size-im_w)
            diff_h = max(0,self.crop_size-im_h)
            padding = (diff_w//2, diff_h//2, diff_w-diff_w//2, diff_h-diff_h//2)
            img = transforms.functional.pad(img, padding, 0, 'constant')
            torch.set_rng_state(state)
            t,l,h,w=transforms.RandomCrop.get_params(img,(self.crop_size,self.crop_size))
            img = transforms.functional.crop(img, t, l, h,w) 
            msk = transforms.functional.pad(msk, padding, 0, 'constant')
            msk = transforms.functional.crop(msk, t, l, h,w)
        img = self.transform_img(img)
        
        if 'all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        else:
            msk = np.array(msk,dtype=int)
            
        mask_cls = np.array(msk==self.cls,dtype=int)
        
        
        # generate mask and prompt
        if self.if_prompt:
            if self.prompt_type =='point':
                prompt,mask_now = get_first_prompt(mask_cls,region_type=self.region_type)
                pc = torch.as_tensor(prompt[:,:2], dtype=torch.float)
                pl = torch.as_tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'point_coords': pc,
                    'point_labels':pl,
                    'img_name':img_path,
            }
            elif self.prompt_type =='box':
                prompt,mask_now = get_top_boxes(mask_cls,region_type=self.region_type)
                box = torch.as_tensor(prompt, dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'boxes':box,
                    'img_name':img_path,
            }
            elif self.prompt_type =='hybrid':
                prompt,mask_now = get_first_prompt(mask_cls,region_type=self.region_type)
                pc = torch.as_tensor(prompt[:,:2], dtype=torch.float)
                pl = torch.as_tensor(prompt[:, -1], dtype=torch.float)
                prompt,mask_now = get_top_boxes(mask_cls,region_type=self.region_type)
                box = torch.as_tensor(prompt, dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'point_coords': pc,
                    'point_labels':pl,
                    'boxes':box,
                    'img_name':img_path,
            }
        else:
            msk = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
            return {'image':img,
                'mask':msk,
                'img_name':img_path,
        }

