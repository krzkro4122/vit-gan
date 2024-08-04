import torch
from torch import nn

from src.v1.config import (
    EncoderParameters,
    MappingMLPParameters,
    TransformerParameters,
    config,
)
from src.v1.patch_encoder import PatchEncoder
from src.v1.transformer import Transformer
from src.v1.muilti_layer_perceptron import MLP
from src.v1.model_utils import count_params


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_encoder = PatchEncoder(encoder_parameters=EncoderParameters())

        discriminator_parameters = config.discriminator_params

        transformer_parameter = TransformerParameters(
            input_features=self.patch_encoder.token_size,
            spectral_scaling=True,
            lp=2,
        )
        self.transformer_layers = nn.ModuleList(
            [
                Transformer(transformer_parameters=transformer_parameter)
                for _ in range(discriminator_parameters.number_of_transformer_layers)
            ]
        )

        self.mlp = MLP(
            mlp_parameters=MappingMLPParameters(
                input_features=self.transformer_layers[-1].input_features,
                output_features=discriminator_parameters.mapping_mlp_params.output_features,
            )
        )
        self.sigmoid = torch.nn.Sigmoid()
        print(f"Discriminator model with {count_params(self)} parameters ready")

    def forward(self, images):
        tokens = self.patch_encoder(images)
        for transformer in self.transformer_layers:
            tokens = transformer(tokens)
        output = self.mlp(
            tokens[:, 0, :]
        )  # we compute the output only with the cls token
        return self.sigmoid(output)
