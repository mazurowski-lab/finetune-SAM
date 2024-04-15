from skimage.measure import label
#Scientific computing 
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.functional import one_hot
import cv2
import random
#Pytorch packages

def random_sum_to(n, num_terms = None):
    '''
    generate num_tersm with sum as n
    '''
    num_terms = (num_terms or r.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[i+1] - a[i] for i in range(len(a) - 1)]



def get_first_prompt(mask_cls,dist_thre_ratio=0.1,prompt_num=5,max_prompt_num=8,region_type='random'):
    '''
    if region_type = random, we random select one region and generate prompt
    if region_type = all, we generate prompt at each object region
    if region_type = largest_k, we generate prompt at largest k region, k <10
    '''
    if prompt_num==-1:
        prompt_num = random.randint(1, max_prompt_num)
    # Find all disconnected regions
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    #print('num of regions found', region_ids)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids+1):
        #find coordinates of points in the region
        binary_msk = np.where(label_msk==region_id, 1, 0)

        # clean some region that is abnormally small
        r = np.sum(binary_msk) / np.sum(mask_cls)
        #print('curr mask over all mask ratio', r)
        ratio_list.append(r)
        regionid_list.append(region_id)
    if len(ratio_list)>0:
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]
    
        if region_type == 'random':
            prompt_num = 1
            regionid_list = [random.choice(regionid_list)] # random choose 1 region
            prompt_num_each_region = [1]
        elif region_type[:7] == 'largest':
            region_max_num = int(region_type[-1])
            #print(region_max_num,prompt_num,len(regionid_list))
            valid_region = min(region_max_num,len(regionid_list))
            if valid_region<prompt_num:
                prompt_num_each_region = random_sum_to(prompt_num,valid_region)
            else:
                prompt_num_each_region = prompt_num*[1]
            regionid_list = regionid_list[:min(valid_region,prompt_num)]
            #print(prompt_num_each_region)


        prompt = []
        mask_curr = np.zeros_like(label_msk)
        

        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk==regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk,mask_curr)


            padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
            dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]

            # sort the distances 
            dist_array=sorted(dist_img.copy().flatten())[::-1]
            dist_array = np.array(dist_array)
            # find the threshold:
            dis_thre = max(dist_array[int(dist_thre_ratio*np.sum(dist_array>0))],1)
            #print(np.max(dist_array))
            #print(dis_thre)
            cY, cX = np.where(dist_img>=dis_thre)
            while prompt_num_each_region[reg_id]>0:
                # random select one prompt
                random_idx = np.random.randint(0, len(cX))
                cx, cy = int(cX[random_idx]), int(cY[random_idx])
                prompt.append((cx,cy,1))
                prompt_num_each_region[reg_id] -=1

        while len(prompt)<max_prompt_num: # repeat prompt to ensure the same size
            prompt.append((cx,cy,1))
    else: # if this image doesn't have target object
        prompt = [(0,0,-1)]
        mask_curr = np.zeros_like(label_msk)
        while len(prompt)<max_prompt_num: # repeat prompt to ensure the same size
            prompt.append((0,0,-1))
    prompt = np.array(prompt) 
    mask_curr = np.array(mask_curr,dtype=int)
    return prompt,mask_curr


def get_top_boxes(mask_cls,dist_thre_ratio=0.1,region_max_num=5,region_type='largest_5'):
    # Find all disconnected regions
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    #print('num of regions found', region_ids)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids+1):
        #find coordinates of points in the region
        binary_msk = np.where(label_msk==region_id, 1, 0)

        # clean some region that is abnormally small
        r = np.sum(binary_msk) / np.sum(mask_cls)
        #print('curr mask over all mask ratio', r)
        ratio_list.append(r)
        regionid_list.append(region_id)
    if len(ratio_list)>0:
        # sort the region from largest to smallest
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]

        if region_type == 'random':
            region_max_num = 1
            regionid_list = [random.choice(regionid_list)] # random choose 1 region
        elif region_type[:7] == 'largest':
            region_max_num = int(region_type[-1])
            regionid_list = regionid_list[:min(region_max_num,len(regionid_list))]

        prompt = []
        mask_curr = np.zeros_like(label_msk)
        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk==regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk,mask_curr)
            box = MaskToBoxSimple(binary_msk,dist_thre_ratio)
            prompt.append(box)

        while len(prompt)<region_max_num: # repeat prompt to ensure the same size
            prompt.append(box)
        prompt = np.array(prompt) 
        mask_curr = np.array(mask_curr,dtype=int)
    else:
        prompt = [[0,0,0,0]]
        mask_curr = np.zeros_like(label_msk)
        while len(prompt)<region_max_num:
            prompt.append(prompt[0])
    return prompt,mask_curr
        
def MaskToBoxSimple(mask,random_thre=0.1):
    '''
    random_thre, the randomness at each side of box
    '''
    mask = mask.squeeze()
    
    y_max,x_max = mask.shape[0],mask.shape[1]
    
    #find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0,x0 = row.min(),col.min()
    y1,x1 = row.max(),col.max()
    
    y_thre = (y1-y0)*random_thre
    x_thre = (x1-x0)*random_thre
    
    x0 = max(0,x0-x_thre*random.random())
    x1 = min(x_max,x1+x_thre*random.random())
    
    y0 = max(0,y0-y_thre*random.random())
    y1 = min(y_max,y1+y_thre*random.random())
    

    return [x0,y0,x1,y1]

