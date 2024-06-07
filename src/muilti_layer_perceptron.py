import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        layers=None,
        activation="gelu",
        dropout_rate=0.0,
        **kwargs
    ):
        """
        Usual MLP module
        :param input_features: number of input features
        :param output_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(MLP, self).__init__()
        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(lp, lnext), nn.Dropout(dropout_rate))
                for lp, lnext in zip(
                    [input_features] + self.layers, self.layers + [output_features]
                )
            ]
        )

        self.activation = (
            torch.nn.ReLU()
            if activation == "relu"
            else (
                torch.nn.Tanh()
                if activation == "tanh"
                else (
                    torch.nn.Sigmoid()
                    if activation == "sigmoid"
                    else torch.nn.GELU() if activation == "gelu" else ValueError
                )
            )
        )

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i != len(self.model) - 1:
                x = self.activation(x)
        return x
