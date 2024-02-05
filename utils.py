import yaml
import os
import numpy as np
from torchvision.models import wide_resnet50_2, resnet18
from data_loader import TrainDataModule
import torch.nn.functional as F
import torch

def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error reading the config file: {e}")
        return None

def load_pretrained_CNN(opt):
    
    model_dict = {
        'resnet18': resnet18,
        'wide_resnet50_2': wide_resnet50_2
    }
    
    # load pretrained CNN
    model_name = opt['model']['backbone']
    t_d = opt['model']['target_dimension']
    r_d = opt['model']['output_dimension']

    model = model_dict[model_name](pretrained=True,  progress=True)
    print('Backbone: {}'.format(opt['model']['backbone']))
    print('Input dim size: {}'.format(t_d))
    print('Output dim size after reduced: {}'.format(r_d))

    return model, t_d, r_d

def load_train_dataset(opt):

    train_data_module = TrainDataModule(
        split_dir=opt['dataset']['ann_path'],
        target_size=opt['dataset']['target_size'],
        batch_size=opt['dataset']['batch_size'])
    
    return train_data_module


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


