import pandas as pd
import os

def _get_train_files(split_dir):
    root_dir = split_dir.split('splits')[0]
    train_csv_ixi = os.path.join(split_dir, 'ixi_normal_train.csv')
    train_csv_fastMRI = os.path.join(split_dir, 'normal_train.csv')
    # Load csv files and split the filename based on './data' and replace it witrh root_dir
    train_files_ixi = pd.read_csv(train_csv_ixi)['filename'].str.replace('./data/', root_dir).tolist()
    train_files_fastMRI = pd.read_csv(train_csv_fastMRI)['filename'].str.replace('./data/', root_dir).tolist()            
    # Combine files
    train_data = train_files_ixi + train_files_fastMRI
    data_dict = {
        'img': train_data,
    }
    return data_dict


def _get_val_files(split_dir):
    root_dir = split_dir.split('splits')[0]
    val_csv = os.path.join(split_dir, 'normal_val.csv')
    val_files = pd.read_csv(val_csv)['filename'].str.replace('./data/', root_dir).tolist()
    # Combine files
    val_data = val_files
    data_dict = {
        'img': val_data,
    }
    return data_dict

def _get_test_files(split_dir, pathology):
    root_dir = split_dir.split('splits')[0]
    img_csv = os.path.join(split_dir, f'{pathology}.csv')
    pos_mask_csv = os.path.join(split_dir, f'{pathology}_ann.csv')
    neg_mask_csv = os.path.join(split_dir, f'{pathology}_neg.csv')

    img_paths = pd.read_csv(img_csv)['filename'].str.replace('./data/', root_dir).tolist()
    pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].str.replace('./data/', root_dir).tolist()
    neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].str.replace('./data/', root_dir).tolist()

    data_dict = {
        'img': img_paths,
        'pos_mask': pos_mask_paths,
        'neg_mask': neg_mask_paths
    }

    return data_dict
