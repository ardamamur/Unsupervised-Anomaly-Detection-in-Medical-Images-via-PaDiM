import os
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class FAST_IXI(Dataset):
    def __init__(self, opt, records, is_train=True, eval_class=None):
        self.opt = opt
        self.dataset_name = opt['dataset']['name']
        self.dataset_path = opt['dataset']['path']
        self.is_train = is_train
        self.resize = self.opt['dataset']['resize']
        self.cropsize = self.opt['dataset']['cropsize']
        self.record_df = records
        self.eval_class = eval_class

        # set transforms
        self.transform_x = T.Compose([T.Resize(self.resize, Image.BICUBIC),
                                      T.CenterCrop(self.cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(self.resize, Image.NEAREST),
                                         T.CenterCrop(self.cropsize),
                                         T.ToTensor()])
        
        # load dataset
        if self.is_train:
            self.x, self.y = self.load_dataset()

        else:
            self.x, self.y, self.full_maps, self.pos_masks, self.neg_masks = self.load_dataset()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        # since only normal images will be used in training, mask image will be all zeros
        if self.is_train:
            mask = torch.zeros(1, self.cropsize, self.cropsize)
            return x, y, mask
        
        # for abnormal images, load the mask image
        else:
            full_map = self.full_maps[index]
            pos_mask = self.pos_masks[index]
            neg_mask = self.neg_masks[index]

            if full_map is not None:
                full_map = Image.open(full_map)
                full_map = self.transform_mask(full_map)
            else:
                full_map = torch.zeros(1, self.cropsize, self.cropsize)

            if pos_mask is not None:
                #print('full_map is not None')
                pos_mask = Image.open(pos_mask)
                pos_mask = self.transform_mask(pos_mask)
            else:
                pos_mask = torch.zeros(1, self.cropsize, self.cropsize)

            if neg_mask is not None:
                neg_mask = Image.open(neg_mask)
                neg_mask = self.transform_mask(neg_mask)
            else:
                neg_mask = torch.zeros(1, self.cropsize, self.cropsize)

            return x, y, full_map, pos_mask, neg_mask
        

    def load_dataset(self):
        phase = 'train' if self.is_train else 'test'
        # get the rows where split is train or test
        records = self.record_df[self.record_df['split'] == phase]        
        # add new column 'label' to dataframe if type contains 'normal' add 0 else add 1
        records['label'] = records['type'].apply(lambda x: 0 if 'normal' in x else 1)

        if self.is_train:
            if self.dataset_name == 'fast':
                records = records[records['type'] == 'normal_fast']
            elif self.dataset_name == 'ixi':
                records = records[records['type'] == 'normal_ixi']
            else:
                pass

            images = records['filename'].tolist()
            labels = records['label'].tolist()
            assert len(images) == len(labels) # sanity check
            return images, labels
        

        else:
            if self.eval_class is not None:
                records = records[records['type'] == self.eval_class]
            else:
                pass

            images = records['filename'].tolist()
            labels = records['label'].tolist()
            full_maps = records['full_map'].tolist()
            pos_masks = records['mask_pos'].tolist()
            neg_masks = records['mask_neg'].tolist()
            assert len(images) == len(labels) == len(pos_masks) == len(neg_masks) == len(full_maps) # sanity check
            return images, labels, full_maps, pos_masks, neg_masks
        