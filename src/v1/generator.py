import torch
from torch import nn

from src.v1.config import MappingMLPParameters, TransformerParameters, config
from src.v1.muilti_layer_perceptron import MLP
from src.v1.transformer import TransformerSLN
from src.v1.spectral_layer_norm import SLN
from src.v1.siren import SIREN, SIRENParameters
from src.v1.model_utils import count_params


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mapping_mlp = MLP(
            mlp_parameters=MappingMLPParameters(
                input_features=config.lattent_space_size,
                output_features=config.image_size
                * config.generator_params.feature_hidden_size,
            )
        )

        self.embedding = torch.nn.Parameter(
            torch.randn(config.image_size, config.generator_params.feature_hidden_size)
        )

        transformer_parameters = TransformerParameters(
            input_features=config.generator_params.feature_hidden_size,
            spectral_scaling=False,
            lp=1,
        )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerSLN(transformer_parameters=transformer_parameters)
                for _ in range(config.generator_params.number_of_transformer_layers)
            ]
        )
        self.sln = SLN(number_of_features=config.generator_params.feature_hidden_size)
        self.output_network = nn.Sequential(
            SIREN(
                siren_parameters=SIRENParameters(
                    input_features=config.generator_params.feature_hidden_size,
                    output_features=config.generator_params.output_hidden_dimension,
                    is_first=True,
                )
            ),
            SIREN(
                siren_parameters=SIRENParameters(
                    input_features=config.generator_params.output_hidden_dimension,
                    output_features=config.number_of_channels * config.image_size,
                    is_first=False,
                )
            ),
        )
        print(f"Generator model with {count_params(self)} parameters ready")

    def forward(self, x):
        weights = self.mapping_mlp(x).view(
            -1, config.image_size, config.generator_params.feature_hidden_size
        )
        h = self.embedding
        for tf in self.transformer_layers:
            weights, h = tf(h, weights)
        weights = self.sln(h, weights)
        result = self.output_network(weights).view(
            x.shape[0], config.number_of_channels, config.image_size, config.image_size
        )
        return result
