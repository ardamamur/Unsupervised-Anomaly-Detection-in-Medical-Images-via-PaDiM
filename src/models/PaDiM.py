""" Pytorch Model for the PaDiM model implementation"""

import torch
import torch.nn.functional as F
from random import sample
from torch import Tensor, nn
from typing import Tuple, Sequence
from utils.environment_settings import env_settings
from models.backbone import FeatureExtractor, get_feature_dims, find_featuremap_dims
from models.gaussian import MultiVariateGaussian
from models.anomaly_map import AnomalyMapGenerator

class PaDiM(nn.Module):
    """
    PaDiM Model
    Args:
        input_size (tuple[int, int]): Input size for the model.
        layers (list[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_features (int, optional): Number of features to retain in the dimension reduction step.
                                Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    """

    def __init__(
            self,
            input_size : Sequence[int],
            layers : list[str],
            backbone : str = "resnet18",
            pre_trained : bool = True,
            #mode : bool = True
    ):
        super().__init__()
        self.backbone = backbone
        #self.mode = mode
        self.input_size = input_size
        self.layers = layers
        self.pre_trained = pre_trained
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone,
            layers = self.layers,
            pre_trained = self.pre_trained
        )

        self.n_features_original, self.n_patches = get_feature_dims(
            self.feature_extractor,
            input_size=self.input_size,
            layers = self.layers
        )

        self.n_features = env_settings._N_FEATURES_DEFAULTS.get(self.backbone)
        
        
        """
        Patch embedding vector may carry redundant information, therefore we experimentally study the possibilty
        to reduce their size. So we noticed that randomly selecting few dimensions.
        So we randomly select n_features from n_features_original
        """
        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(0, self.n_features_original), self.n_features)),
        )
        self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)


    def forward(self, input_tensor: Tensor):
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Example:
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = self.extract_features(input_tensor)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])

            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)
        
        if self.training:
            output = embeddings

        else:
            output = self.anomaly_map_generator(
                embedding=embeddings,
                mean = self.gaussian.mean,
                inv_covariance = self.gaussian.inv_covariance
            )
        
        return output
    

    def generate_embedding(self, features):
        """Generate embedding from hierarchical feature map.

        Args:
            features (dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings
