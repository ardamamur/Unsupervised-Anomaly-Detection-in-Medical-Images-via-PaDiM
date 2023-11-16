from typing import Sequence, Optional
import pytorch_lightning as pl
import json
from src.utils._preprare_data import _get_train_files, _get_test_files, _get_val_files
from src.data.brain_mri_dataset import BrainMRI
from torch.utils.data import DataLoader

class BrainMriLightning(pl.LightningDataModule):
    def __init__(self, batch_size: int, crop_size: Sequence[int], split_dir:str, pathology: Optional[str] ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.split_dir = split_dir
        self.pathology = pathology

    def setup(self, stage: str = None) -> None:
        # This function is called on every GPU when you do model.fit()
        # It is used to load the data from the disk
        # This function is called on every GPU when you do model.fit()
        # It is used to load the data from the disk

        if stage == 'fit' or stage is None:
            train_data = _get_train_files(self.split_dir)
            val_data = _get_val_files(self.split_dir)
            self.train_dataset = BrainMRI(data = train_data, crop_size = self.crop_size, mode = 'train')
            self.val_dataset = BrainMRI(data = val_data, crop_size = self.crop_size, mode = 'val')

        if stage == 'test' or stage is None:
            test_data = _get_test_files(self.split_dir, self.pathology)
            print(test_data)
            #assert len(test_data['img']) == len(test_data['pos_mask']) == len(test_data['neg_mask']), 'Length of the data is not equal'
            self.test_dataset = BrainMRI(data = test_data, crop_size = self.crop_size, mode = 'test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=16, pin_memory=True)
    
    def val_dataloader(self) -> DataLoader:
        # This function returns the dataloader for validation
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=16, pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        # This function returns the dataloader for testing
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=16, pin_memory=True)
    