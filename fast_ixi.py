import os
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class FAST_IXI(Dataset):
    def __init__(self, opt, is_train=True):
        self.opt = opt
        self.dataset_path = opt['dataset']['path']
        self.is_train = is_train
        self.resize = self.opt['dataset']['resize']
        self.cropsize = self.opt['dataset']['cropsize']

        # set transforms
        self.transform_x = T.Compose([T.Resize(self.resize, Image.ANTIALIAS),
                                      T.CenterCrop(self.cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(self.resize, Image.NEAREST),
                                         T.CenterCrop(self.cropsize),
                                         T.ToTensor()])
        
        # load dataset

        self.x, self.y, self.mask = self.load_dataset()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        mask = self.mask[index]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        # since only normal images will be used in training, mask image will be all zeros
        if y == 0:
            mask = torch.zeros(1, self.cropsize, self.cropsize)
        else:
            if mask is not None:
                mask = Image.open(mask)
                mask = self.transform_mask(mask)
            else:
                mask = torch.zeros(1, self.cropsize, self.cropsize)

        return x, y, mask

    def load_dataset(self):
        phase = 'train' if self.is_train else 'val'
        x, y, mask = [], [], []
        
        img_dir = os.path.join(self.dataset_path, phase)
        img_types = os.listdir(img_dir)
        mask_dir = os.path.join(self.dataset_path, 'mask')

        # train -> normal 
        # val -> normal, abnormal

        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                                for f in os.listdir(img_type_dir)
                                                if f.endswith('.png')])
            
            x.extend(img_fpath_list)

            # load labels
            if img_type == 'normal':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                mask_type_dir = os.path.join(mask_dir)
                mask_fpath_list = sorted([os.path.join(mask_type_dir, f) 
                                                for f in os.listdir(img_type_dir)
                                                if f.endswith('.png')])
                mask.extend(mask_fpath_list)


        assert len(x) == len(y) # sanity check

        return list(x), list(y), list(mask)