import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
import numpy as np  
import torch
import torch.nn.functional as F


class PaDiM():
    def __init__(self, opt, model, train_dataloader, outputs, idx, device):
        self.opt = opt
        self.model = model
        self.train_dataloader = train_dataloader
        self.outputs = outputs
        self.idx = idx
        self.device = device
        self.train_feature_filepath = opt['model']['train_feature_filepath']
        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    def embedding_concat(self, x, y):
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

    def extract_features(self):
        print('Extracting features from train dataset...')

        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        

        # for each batch in the dataloader (use tqdm bar), train dataloader get item returns only x
        for batch_idx, img in tqdm(enumerate(self.train_dataloader), '| feature extraction | train | %s |' % 'brainmri'):
            img = img.to(self.device)
            with torch.no_grad():
                _ = self.model(img)
            for key, value in zip(self.train_outputs.keys(), outputs):
                self.train_outputs[key].append(value.cpu().detach())
            # initialize hook outputs
            outputs = []

        for key, value in self.train_outputs.items():
            self.train_outputs[key] = torch.cat(value, 0)

        print('first layer shape:', self.train_outputs['layer1'].shape)
        print('second layer shape:', self.train_outputs['layer2'].shape)
        print('third layer shape:', self.train_outputs['layer3'].shape)
        # Embedding concat
        embedding_vectors = self.train_outputs['layer1'] # get the maximum size of the embedding vectors
    
        """
        Rresearchers conceptually divide the input image into a grid based on the resolution of the largest activation map—typically
        the first layer of the pre-trained CNN. This way, each grid position, denoted as (i,j), 
        is associated with a unique embedding vector that represents the collective activation vectors for that particular image patch.
        """
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, self.train_outputs[layer_name])

        # randomly select d dimension
        print('randomly select %d dimension' % self.opt['model']['output_dimension'])
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        B, C, H, W = embedding_vectors.size() # Get the shape of the embedding vectors which is same with the first layer of the pretrained model
        print('embedding_vectors shape:', embedding_vectors.shape)
        embedding_vectors = embedding_vectors.view(B, C, H * W)

        # calculate multivariate Gaussian distribution
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)

        # calculate mean, cov and inverse covariance matrix for each patch position at Xij 
        # (each patch position (i,j) is associated with a unique embedding vector)
        for i in range(H * W):
            # Xij = embedding_vectors[:, :, i].numpy()
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I

        # save learned distribution
        self.train_outputs = [mean, cov]

    def save_extracted_features(self):
        print('Saving extracted features...')
        with open(self.train_feature_filepath, 'wb') as f:
            pickle.dump(self.train_outputs, f)

    def load_extracted_features(self):
        print('Loading extracted features...')
        with open(self.train_feature_filepath, 'rb') as f:
            self.train_outputs = pickle.load(f)
