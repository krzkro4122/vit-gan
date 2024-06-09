import torch
from torch import nn

from patch_encoder import PatchEncoder
from transformer import Transformer
from muilti_layer_perceptron import MLP
from model_utils import count_params


class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        number_of_channels,
        output_size,
        number_of_transformer_layers=1,
        encoder_params=None,
        transformer_params=None,
        mlp_params=None,
        **kwargs,
    ):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.number_of_channels = number_of_channels
        self.output_size = output_size
        self.number_of_transformer_layers = number_of_transformer_layers

        self.encoder_params = {} if encoder_params is None else encoder_params
        self.transformer_params = (
            {} if transformer_params is None else transformer_params
        )
        self.mlp_params = {} if mlp_params is None else mlp_params

        self.encoder_params["image_size"], self.encoder_params["number_of_channels"] = (
            self.image_size,
            self.number_of_channels,
        )
        self.patch_encoder = PatchEncoder(**self.encoder_params)

        (
            self.transformer_params["input_features"],
            self.transformer_params["spectral_scaling"],
            self.transformer_params["lp"],
        ) = (self.patch_encoder.projection_output_size, True, 2)
        self.transformer_layers = nn.ModuleList(
            [
                Transformer(**self.transformer_params)
                for _ in range(self.number_of_transformer_layers)
            ]
        )

        self.mlp_params["input_features"], self.mlp_params["output_features"] = (
            self.transformer_layers[-1].input_features,
            self.output_size,
        )
        self.mlp = MLP(**self.mlp_params)

        self.sigmoid = torch.nn.Sigmoid()

        print(f"Discriminator model with {count_params(self)} parameters ready")

    def forward(self, imgs):
        tokens = self.patch_encoder(imgs)
        for transformer in self.transformer_layers:
            tokens = transformer(tokens)
        output = self.mlp(
            tokens[:, 0, :]
        )  # we compute the output only with the cls token
        return self.sigmoid(output)
