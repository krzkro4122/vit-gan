import torch
from torch import nn

from src.config import MappingMLPParameters


def pick_activation(activation: str):
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "gelu":
        return torch.nn.GELU()
    if activation == "tanh":
        return torch.nn.Tanh()
    else:
        return torch.nn.Sigmoid()


class MLP(nn.Module):
    def __init__(
        self,
        mlp_parameters: MappingMLPParameters,
    ):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(lp, lnext), nn.Dropout(mlp_parameters.dropout_rate)
                )
                for lp, lnext in zip(
                    [mlp_parameters.input_features] + mlp_parameters.layers,
                    mlp_parameters.layers + [mlp_parameters.output_features],
                )
            ]
        )
        self.activation = pick_activation(mlp_parameters.activation)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i != len(self.model) - 1:
                x = self.activation(x)
        return x
