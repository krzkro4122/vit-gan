import numpy as np
from pydantic import BaseModel
import torch
from torch import nn


class SIRENParameters(BaseModel):
    input_features: int
    output_features: int
    bias: bool = True
    is_first: bool = False
    omega_0: int = 30


class SIREN(nn.Module):
    def __init__(
        self,
        siren_parameters: SIRENParameters,
    ):
        self.siren_parameters = siren_parameters
        super().__init__()
        self.linear = nn.Linear(
            self.siren_parameters.input_features,
            self.siren_parameters.output_features,
            bias=self.siren_parameters.bias,
        )
        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            if self.siren_parameters.is_first:
                self.linear.weight.uniform_(
                    -1 / self.siren_parameters.input_features,
                    1 / self.siren_parameters.input_features,
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.siren_parameters.input_features)
                    / self.siren_parameters.omega_0,
                    np.sqrt(6 / self.siren_parameters.input_features)
                    / self.siren_parameters.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.siren_parameters.omega_0 * self.linear(x))
