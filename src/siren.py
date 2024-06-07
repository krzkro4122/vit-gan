import numpy as np
import torch
from torch import nn


class SIREN(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        is_first=False,
        omega_0=30,
        **kwargs
    ):
        """
        Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN
        :param input_features: number of input features
        :param out_features: number of output features
        :param bias: add a bias or not to the linear transformation
        :param is_first: first layer
        :param omega_0: pulsation of the sine activation
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = input_features
        self.linear = nn.Linear(input_features, output_features, bias=bias)

        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.input_features, 1 / self.input_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.input_features) / self.omega_0,
                    np.sqrt(6 / self.input_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
