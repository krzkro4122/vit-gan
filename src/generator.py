import torch
from torch import nn

from muilti_layer_perceptron import MLP
from transformer import TransformerSLN
from spectral_layer_norm import SLN
from siren import SIREN

from model_utils import count_params


class Generator(nn.Module):
    def __init__(
        self,
        lattent_size,
        image_size,
        number_of_channels,
        feature_hidden_size=384,
        number_of_transformer_layers=1,
        output_hidden_dimension=768,
        mapping_mlp_params=None,
        transformer_params=None,
        **kwargs,
    ):
        """
        ViT Generator Class
        :param lattent_size: number of features in the lattent space
        :param image_size: output images size, the image will be square sized
        :param number_of_channels: number of channel in the output images
        :param feature_hidden_size: number of features in the transformers and output layers
        :umber_ofparam n_transformer_layers: number of stacked transformer blocks
        :param mapping_mlp_params: kwargs for optional parameters of the mapping MLP, mandatory args will be filled automatically
        :param transformer_params: kwargs for optional parameters of the Transformer blocks, mandatory args will be filled automatically
        """
        super(Generator, self).__init__()

        self.lattent_size = lattent_size
        self.image_size = image_size
        self.feature_hidden_size = feature_hidden_size
        self.number_of_channels = number_of_channels
        self.number_of_transformer_layers = number_of_transformer_layers
        self.output_hidden_dimension = output_hidden_dimension

        self.mapping_params = {} if mapping_mlp_params is None else mapping_mlp_params
        self.transformer_params = (
            {} if transformer_params is None else transformer_params
        )

        self.mapping_params["in_features"], self.mapping_params["out_features"] = (
            self.lattent_size,
            self.image_size * self.feature_hidden_size,
        )
        self.mapping_mlp = MLP(**self.mapping_params)

        self.emb = torch.nn.Parameter(
            torch.randn(self.image_size, self.feature_hidden_size)
        )

        (
            self.transformer_params["in_features"],
            self.transformer_params["spectral_scaling"],
            self.transformer_params["lp"],
        ) = (self.feature_hidden_size, False, 1)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerSLN(**self.transformer_params)
                for _ in range(self.number_of_transformer_layers)
            ]
        )

        self.sln = SLN(self.feature_hidden_size)

        self.output_net = nn.Sequential(
            SIREN(self.feature_hidden_size, output_hidden_dimension, is_first=True),
            SIREN(
                output_hidden_dimension,
                self.number_of_channels * self.image_size,
                is_first=False,
            ),
        )

        print(f"Generator model with {count_params(self)} parameters ready")

    def forward(self, x):
        weights = self.mapping_mlp(x).view(
            -1, self.image_size, self.feature_hidden_size
        )
        h = self.emb
        for tf in self.transformer_layers:
            weights, h = tf(h, weights)
        weights = self.sln(h, weights)
        result = self.output_net(weights).view(
            x.shape[0], self.number_of_channels, self.image_size, self.image_size
        )
        return result
