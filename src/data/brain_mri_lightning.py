from typing import Sequence, Tuple, Optional
import pytorch_lightning as pl
import json
from utils._preprare_data import _get_train_files, _get_test_files, _get_val_files
from data.brain_mri_dataset import BrainMRI_Train, BrainMRI_Eval
from torch.utils.data import DataLoader

class BrainMri_Train_Lightning(pl.LightningDataModule):
    def __init__(self, batch_size: int, crop_size: Sequence[int], split_dir:str ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.split_dir = split_dir

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            train_data = _get_train_files(self.split_dir)
            val_data = _get_val_files(self.split_dir)
            self.train_dataset = BrainMRI_Train(data = train_data, crop_size = self.crop_size, mode = 'train')
            self.val_dataset = BrainMRI_Train(data = val_data, crop_size = self.crop_size, mode = 'val')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=16, pin_memory=True)
    
    def val_dataloader(self) -> DataLoader:
        # This function returns the dataloader for validation
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=16, pin_memory=True)

    
class BrainMri_Eval_Lightning(pl.LightningDataModule):
    def __init__(self, batch_size: int, crop_size: Sequence[int], split_dir:str) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.split_dir = split_dir
        self.pathologies = [
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
            'other'
        ]

    def setup(self, stage: str = None) -> None:
        if stage == 'test' or stage is None:
            test_dataset = {}
            for pathology in self.pathologies:
                test_data = _get_test_files(self.split_dir, pathology)
                test_dataset[pathology] = BrainMRI_Eval(data = test_data, crop_size = self.crop_size)
            self.test_dataset = test_dataset


    def test_dataloader(self) -> DataLoader:
        # This function returns the dataloader for validation
        loader_dict = {}
        for pathology in self.test_dataset.keys():
            loader_dict[pathology] = DataLoader(self.test_dataset[pathology], batch_size=self.batch_size,
                                    shuffle=False, num_workers=16, pin_memory=True)
            
        return loader_dict