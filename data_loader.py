import os
from typing import List, Tuple
import torch
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class TrainDataset(Dataset):

    def __init__(self, data: List[str], target_size=(128, 128)):
        """
        Loads images from data

        @param data:
            paths to images
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        """
        super(TrainDataset, self).__init__()
        self.target_size = target_size
        self.data = data

        self.transform = T.Compose([T.Resize(self.target_size, Image.BICUBIC),
                                      T.CenterCrop(self.target_size),
                                      T.Pad(((self.target_size[1] - self.target_size[0]) // 2, 0), fill=0),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.data[idx]).convert('RGB')
        # Pad to square
        #
        # Resize
        #img = img.resize(self.target_size, Image.BICUBIC)
        # Convert to tensor
        #img = transforms.ToTensor()(img)
        # create labels : 0 for normal, 1 for abnormal
        img = self.transform(img)
        #img = T.Pad(((img.height - img.width) // 2, 0), fill=0)(img)
        return img


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, split_dir: str, target_size=(128, 128), batch_size: int = 32):
        """
        Data module for training

        @param split_dir: str
            path to directory containing the split files
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        @param: batch_size: int, default: 32
            batch size
        """
        super(TrainDataModule, self).__init__()
        self.target_size = target_size
        self.batch_size = batch_size

        train_csv_ixi = os.path.join(split_dir, 'ixi_normal_train.csv')
        train_csv_fastMRI = os.path.join(split_dir, 'normal_train.csv')
        val_csv = os.path.join(split_dir, 'normal_val.csv')

        # Load csv files
        train_files_ixi = pd.read_csv(train_csv_ixi)['filename'].tolist()
        train_files_fastMRI = pd.read_csv(train_csv_fastMRI)['filename'].tolist()
        val_files = pd.read_csv(val_csv)['filename'].tolist()

        # Combine files
        self.train_data = train_files_ixi + train_files_fastMRI
        self.val_data = val_files

        # Logging
        print(f"Using {len(train_files_ixi)} IXI images "
              f"and {len(train_files_fastMRI)} fastMRI images for training. "
              f"Using {len(val_files)} images for validation.")

    def train_dataloader(self):
        return DataLoader(TrainDataset(self.train_data, self.target_size),
                          batch_size=self.batch_size,
                          #shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(TrainDataset(self.val_data, self.target_size),
                          batch_size=self.batch_size,
                          shuffle=False)


class TestDataset(Dataset):

    def __init__(self, isNormal:bool, img_csv: str, pos_mask_csv: str, neg_mask_csv: str, target_size=(128, 128)):
        """
        Loads anomalous images, their positive masks and negative masks from data_dir

        @param img_csv: str
            path to csv file containing filenames to the images
        @param img_csv: str
            path to csv file containing filenames to the positive masks
        @param img_csv: str
            path to csv file containing filenames to the negative masks
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        """
        super(TestDataset, self).__init__()
        self.target_size = target_size
        self.isNormal = isNormal
        self.img_paths = pd.read_csv(img_csv)['filename'].tolist()
        self.pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].tolist()
        if not self.isNormal:
            self.neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].tolist()
            assert len(self.img_paths) == len(self.pos_mask_paths) == len(self.neg_mask_paths)
        else:
            assert len(self.img_paths) == len(self.pos_mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = img.resize(self.target_size, Image.BICUBIC)
        img = T.ToTensor()(img)
        # Load positive mask
        pos_mask = Image.open(self.pos_mask_paths[idx]).convert('L')
        pos_mask = pos_mask.resize(self.target_size, Image.NEAREST)
        pos_mask = T.ToTensor()(pos_mask)

        if self.isNormal:
            label = 0
            pos_mask = torch.zeros(1, self.target_size[0], self.target_size[1])
            neg_mask = torch.zeros(1, self.target_size[0], self.target_size[1])
            
        else:
            label = 1
            # Load negative mask
            neg_mask = Image.open(self.neg_mask_paths[idx]).convert('L')
            neg_mask = neg_mask.resize(self.target_size, Image.NEAREST)
            neg_mask = T.ToTensor()(neg_mask)

        return img, label, pos_mask, neg_mask


def get_test_dataloader(split_dir: str, pathology: str, target_size: Tuple[int, int], batch_size: int):
    """
    Loads test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param pathology: str
        pathology to load
    @param batch_size: int
        batch size
    """

    if pathology == 'normal':
        img_csv = os.path.join(split_dir, 'normal_test.csv')
        pos_mask_csv = os.path.join(split_dir, f'{pathology}_test_ann.csv')
        neg_mask_csv = None
        isNormal = True
    else:
        img_csv = os.path.join(split_dir, f'{pathology}.csv')
        pos_mask_csv = os.path.join(split_dir, f'{pathology}_ann.csv')
        neg_mask_csv = os.path.join(split_dir, f'{pathology}_neg.csv')
        isNormal = False

    return DataLoader(TestDataset(isNormal, img_csv, pos_mask_csv, neg_mask_csv, target_size),
                      batch_size=batch_size,
                      shuffle=False,
                      drop_last=False)


def get_all_test_dataloaders(split_dir: str, target_size: Tuple[int, int], batch_size: int):
    """
    Loads all test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param batch_size: int
        batch size
    """

    pathologies = [
        'absent_septum',
        'artefacts',
        'craniatomy',
        'dural',
        'ea_mass',
        'edema',
        'encephalomalacia',
        'enlarged_ventricles',
        'intraventricular',
        'lesions',
        'mass',
        'posttreatment',
        'resection',
        'sinus',
        'wml',
        'other',
        'normal'
    ]
    return {pathology: get_test_dataloader(split_dir, pathology, target_size, batch_size)
            for pathology in pathologies}