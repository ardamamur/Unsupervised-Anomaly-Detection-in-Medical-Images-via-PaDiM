import os
import random
from random import sample
import numpy as np  
import torch
from utils import read_config, load_pretrained_CNN, load_train_dataset
from padim import PaDiM

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'Using device: {device}')

def main():
    # read config file
    config_path = 'config.yaml'
    opt = read_config(config_path)
    experiment_path = opt['dataset']['save_dir'] + '/' + opt['model']['backbone']
    opt['model']['experiment_path'] = experiment_path
    
    os.makedirs(os.path.join(experiment_path, 'temp_%s' % opt['model']['backbone']), exist_ok=True)
    train_feature_filepath = os.path.join(experiment_path, 'temp_%s' % opt['model']['backbone'], 'train_%s.pkl' % 'brainmri')
    opt['model']['train_feature_filepath'] = train_feature_filepath

    # set random seed
    seed = opt['model']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


    # Load the pretrained CNN
    model, t_d, r_d = load_pretrained_CNN(opt)
    model = model.to(device)
    model.eval()

    # select randomly choosen dimension to reduce dimensionality of the feature vector (like PCA)
    idx = torch.tensor(sample(range(0, t_d), r_d))
    
    # set the model's intermediate outputs
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    
    train_dataloader = load_train_dataset(opt).train_dataloader()
    padim = PaDiM(opt, model, train_dataloader, outputs, idx, device)
   
    if not os.path.exists(train_feature_filepath):
        padim.extract_features()
        padim.save_extracted_features()
    else:
        padim.load_extracted_features()

    learned_distribution = padim.train_outputs

    return

if __name__ == '__main__':
    main()
