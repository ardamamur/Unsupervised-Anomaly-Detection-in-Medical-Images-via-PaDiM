import os 
import torch
from typing import Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchio as tio
from monai import transforms as monai_transforms
from pathlib import Path
import numpy as np
import pandas as pd

class BrainMRI(Dataset):
    def __init__(self, data, crop_size: Tuple[int, int], mode: str) -> None:
        super().__init__()

        self.crop_size = crop_size
        self.data = data
        self.mode = mode
        self.transformations = self._get_transformations()
        self.test_transformations = self._get_test_transformations()

    def __len__(self) -> int:
        # This function is called when you do len(instance) on your dataset instance
        return len(self.data['img'])
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # This function is called when you do dataset[index] on your dataset instance
        print(self.data['img'][index])
        img = Image.open(self.data['img'][index]).convert('L')
        
        if self.mode == 'train':
            img = self.transformations(img)
            return img
        
        elif self.mode == 'val':
            img = self.test_transformations(img)
            return img
        
        elif self.mode == 'test':
            img = self.test_transformations(img)
            pos_mask = Image.open(self.data['pos_mask'][index]).convert('L')
            neg_mask = Image.open(self.data['neg_mask'][index]).convert('L')
            pos_mask = self.test_transformations(pos_mask)
            neg_mask = self.test_transformations(neg_mask)
            return img, pos_mask, neg_mask
        
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        

    def _get_transformations(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomResizedCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])
    
    def _get_test_transformations(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
        ])