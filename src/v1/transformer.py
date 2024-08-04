from torch import nn

from src.v1.config import MappingMLPParameters, TransformerParameters
from src.v1.attention import MultiHeadSelfAttention
from src.v1.spectral_layer_norm import SLN
from src.v1.muilti_layer_perceptron import MLP


class Transformer(nn.Module):
    def __init__(self, transformer_parameters: TransformerParameters):
        super().__init__()

        self.head_output_dimension = (
            transformer_parameters.input_features
            // transformer_parameters.number_of_heads
        )

        self.layer_norm_1 = nn.LayerNorm(transformer_parameters.input_features)
        self.layer_norm_2 = nn.LayerNorm(transformer_parameters.input_features)

        self.attention_dropout = nn.Dropout(
            transformer_parameters.attention_dropout_rate
        )

        self.msha = MultiHeadSelfAttention(
            transformer_parameters=transformer_parameters,
            head_dimension=self.head_output_dimension,
            output_size=transformer_parameters.input_features,
        )
        self.mlp = MLP(
            mlp_parameters=MappingMLPParameters(
                input_features=transformer_parameters.input_features,
                output_features=transformer_parameters.input_features,
                dropout_rate=transformer_parameters.mlp_dropout,
                layers=transformer_parameters.mlp_layers,
                activation=transformer_parameters.mlp_activation,
            )
        )

    def forward(self, x):
        x1 = self.layer_norm_1(x)
        x = x + self.attention_dropout(self.msha(x1))
        x2 = self.layer_norm_2(x)
        out = x + self.mlp(x2)
        return out


class TransformerSLN(nn.Module):
    def __init__(
        self,
        transformer_parameters: TransformerParameters,
    ):
        super().__init__()
        self.head_output_dimension = (
            transformer_parameters.input_features
            // transformer_parameters.number_of_heads
        )

        self.layer_norm_1 = SLN(
            number_of_features=transformer_parameters.input_features
        )
        self.layer_norm_2 = SLN(
            number_of_features=transformer_parameters.input_features
        )

        self.attention_dropout = nn.Dropout(
            transformer_parameters.attention_dropout_rate
        )

        self.msha = MultiHeadSelfAttention(
            transformer_parameters=transformer_parameters,
            head_dimension=self.head_output_dimension,
            output_size=transformer_parameters.input_features,
        )
        self.mlp = MLP(
            mlp_parameters=MappingMLPParameters(
                input_features=transformer_parameters.input_features,
                output_features=transformer_parameters.input_features,
                layers=transformer_parameters.mlp_layers,
                activation=transformer_parameters.mlp_activation,
                dropout_rate=transformer_parameters.mlp_dropout,
            )
        )

    def forward(self, h, x):
        htmp = self.attention_dropout(self.msha(self.layer_norm_1(h, x))) + h
        hf = self.mlp(self.layer_norm_2(htmp, x)) + htmp
        return x, hf
