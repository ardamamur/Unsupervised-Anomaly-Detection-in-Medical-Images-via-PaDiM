from typing import Tuple, Sequence

import torch
import timm
from torch import nn, Tensor
from torchvision.models import resnet18, resnet50, wide_resnet50_2

import warnings

class FeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
            Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
            computation is required.

    Example:
    The number of features and resolution at each layer ("layer1", "layer2", "layer3") 
    of a ResNet-like model can be determined based on the structure of the network and the input size. 
    Given an input size of (1, 3, 128, 128), which corresponds to (batch size, channels, height, width),
    we can analyze each layer:

    1. **Layer1**: 
    - This layer contains BasicBlocks with convolutions of stride 1 and padding 1, 
    so the spatial dimensions (height and width) of the feature maps do not change in this layer.
    - Since the first convolutional layer (`conv1`) in the network has a stride of 2 and is followed 
    by a max pooling with stride 2, the spatial dimensions are reduced 
    by a factor of 4 before entering "layer1". Therefore, for an input of size 128x128, 
    the feature map size at "layer1" would be 32x32.
    - The number of features (channels) output by "layer1" is determined
    by the output channels of its convolutions. In this case, 
    the convolutions in "layer1" output 64 channels.

    2. **Layer2**: 
    - In "layer2", the first BasicBlock includes a convolution with stride 2, 
    which halves the spatial dimensions. 
    The subsequent convolutions in "layer2" maintain the spatial dimensions.
    - Therefore, the feature map size at "layer2" would be 16x16 (half of 32x32).
    - The number of features output by "layer2" is determined by its convolutions,
    which output 128 channels.

    3. **Layer3**: 
    - Similar to "layer2", the first BasicBlock in "layer3" includes a convolution with stride 2, 
    halving the spatial dimensions again.
    - Thus, the feature map size at "layer3" would be 8x8.
    - The convolutions in "layer3" output 256 channels, 
    so that's the number of features at this layer.

    In summary, for an input of size (1, 3, 128, 128), the features and resolutions are:
    - **Layer1**: 64 features, 32x32 resolution
    - **Layer2**: 128 features, 16x16 resolution
    - **Layer3**: 256 features, 8x8 resolution

    These are based on the standard architecture of ResNet 
    """

    def __init__(self, backbone: str, layers: list[str], pre_trained: bool = True, requires_grad: bool = False):
        super().__init__()

        # Extract backbone-name and weight-URI from the backbone string.
        if "__AT__" in backbone:
            backbone, uri = backbone.split("__AT__")
            pretrained_cfg = timm.models.registry.get_pretrained_cfg(backbone)
            # Override pretrained_cfg["url"] to use different pretrained weights.
            pretrained_cfg["url"] = uri
        else:
            pretrained_cfg = None

        self.backbone = backbone
        self.layers = layers
        self.idx = self._map_layer_to_idx()
        self.requires_grad = requires_grad
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            pretrained_cfg=pretrained_cfg,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.out_dims = self.feature_extractor.feature_info.channels()
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self, offset: int = 3) -> list[int]:
        """Maps set of layer names to indices of model.

        Args:
            offset (int) `timm` ignores the first few layers when indexing please update offset based on need

        Returns:
            Feature map extracted from the CNN
        """
        idx = []
        features = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=False,
            exportable=True,
        )
        for i in self.layers:
            try:
                idx.append(list(dict(features.named_children()).keys()).index(i) - offset)
            except ValueError:
                warnings.warn(f"Layer {i} not found in model {self.backbone}")
                # Remove unfound key from layer dict
                self.layers.remove(i)

        return idx

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        if self.requires_grad:
            features = dict(zip(self.layers, self.feature_extractor(inputs)))
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = dict(zip(self.layers, self.feature_extractor(inputs)))
        return features





def find_featuremap_dims(
        feature_extractor: FeatureExtractor,
        input_size: Sequence[int],
        layers: list[str]
        ):
    """
    Run an empty input tensor to get the feature map tensors' dimensions
    (num_features, resolution)

    As activation maps have a lower resolution than the input image,many pixels have the same embeddings 
    and then form pixel patches with no overlap in the original image resolution. Hence  an input image
    can be divided into a frid of (i,j) = (1,W) * (1,H) positions, where W an H is the resolution of the 
    largest activation map used to generate embeddings. 
    
    In this setup first layer of the pretrained network has the largest resolution
    
    Returns:
        tuple[int, int]: maping of `layer -> dimensions dict`
        Each `dimension dict` has two keys: `num_features` (int) and `resolution`(tuple[int, int]).
    """

    input_tensor = torch.empty(1,3,*input_size)
    input_features = feature_extractor(input_tensor)

    return {
        layer: {"num_features": input_features[layer].shape[1], "resolution": input_features[layer].shape[2:]}
        for layer in layers
    }



def get_feature_dims(
        feature_extractor : FeatureExtractor,
        input_size : Sequence[int],
        layers : list[str],
    ) -> tuple[int, int]:

        """Run feature extractor with empty input where size is equal to original image size to deduce the dimensions of the extracted features.

        Important: `layers` is assumed to be ordered and the first (layers[0])
                    is assumed to be the layer with largest resolution.

        Returns:
            tuple[int, int]: Dimensions of the extracted features: (n_dims_original, n_patches)
        """

        dimensions_mapping = find_featuremap_dims(
            feature_extractor=feature_extractor,
            input_size=input_size,
            layers=layers
        )

        # the first layer in `layers` has the largest resolution
        largest_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
        # TODO : explain the line below
        # 
        n_patches = torch.tensor(largest_layer_resolution).prod().int().item()

        # the original embedding size is the sum of the channels of all layers
        n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)  # type: ignore

        return n_features_original, n_patches 