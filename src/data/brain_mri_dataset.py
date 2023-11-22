import os 
import torch
from typing import Optional, Tuple, Sequence
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchio as tio
from monai import transforms as monai_transforms
from pathlib import Path
import numpy as np
import pandas as pd




class BrainMRI_Train(Dataset):
    def __init__(self, data, crop_size: Sequence[int], mode:str) -> None:
        super().__init__()

        self.crop_size = crop_size
        self.data = data
        self.mode = mode
        self.height = 128
        self.width = 128
        self.val_transformations = self._get_val_transformations()
        self.transformations = self._get_transformations()

    def __len__(self) -> int:
        # This function is called when you do len(instance) on your dataset instance
        return len(self.data['img'])
    
    def __getitem__(self, index: int):
        # This function is called when you do dataset[index] on your dataset instance
        # print(self.data['img'][index])
        # LOad the image
        img = Image.open(self.data['img'][index]).convert('RGB')
        #img = transforms.Pad(((img.height - img.width) // 2, 0), fill=0)(img)
        #img = img.resize(self.crop_size, Image.BICUBIC)

        #self.height = img.height
        #self.width = img.width

        if self.mode == "val":
            img = self.val_transformations(img)
        else:        
            img = self.transformations(img)
        return img
    
    def _get_transformations(self) -> transforms.Compose:

        return transforms.Compose([
            #transforms.Pad(((self.height - self.width) // 2, 0), fill=0),
            transforms.RandomResizedCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
        ])
    

    def _get_val_transformations(self) -> transforms.Compose:

        return transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
        ])


class BrainMRI_Eval(Dataset):
    def __init__(self, data, crop_size: Sequence[int])-> None:
        super().__init__()

        self.crop_size = crop_size
        self.data = data
        self.transformations = self._get_transformations()

    def __len__(self) -> int:
        # This function is called when you do len(instance) on your dataset instance
        return len(self.data['img'])
    
    def __getitem__(self, index: int):
        # This function is called when you do dataset[index] on your dataset instance
        # print(self.data['img'][index])
        # LOad the image
        img = Image.open(self.data['img'][index]).convert('RGB')
        pos_mask = Image.open(self.data['pos_mask'][index]).convert('RGB')
        neg_mask = Image.open(self.data['neg_mask'][index]).convert('RGB')

        volumes = np.stack([np.array(img), np.array(pos_mask), np.array(neg_mask)])

        volumes = self.transformations(volumes)

        img = volumes[0]
        pos_mask = volumes[1]
        neg_mask = volumes[2]

        return img, pos_mask, neg_mask
    
    
    def _get_transformations(self) -> transforms.Compose:

        return transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
        ])