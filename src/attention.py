import torch
from torch import nn

from src.config import TransformerParameters


class Attention(nn.Module):
    def __init__(
        self,
        transformer_parameters: TransformerParameters,
        output_features,
        scale=None,
    ):
        super().__init__()
        self.output_features = output_features
        self.scale = output_features if scale is None else scale
        self.spectral_scaling = transformer_parameters.spectral_scaling

        assert transformer_parameters.lp in [
            1,
            2,
        ], f"Unsupported norm for attention: lp={transformer_parameters.lp} but should be 1 or 2"
        self.attention_func = self._l1att if transformer_parameters.lp == 1 else self._l2att

        self.q = nn.Linear(
            transformer_parameters.input_features, self.output_features, bias=False
        )
        self.k = nn.Linear(
            transformer_parameters.input_features, self.output_features, bias=False
        )
        self.v = nn.Linear(
            transformer_parameters.input_features, self.output_features, bias=False
        )

        if self.spectral_scaling:
            sq, sk, sv = self._get_spectrum()
            self.init_spectrum = [max(sq), max(sk), max(sv)]

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        if self.spectral_scaling:
            self._weight_spectral_rescale()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attention = self.attention_func(q, k)  # we use L2 reg only in discriminator
        attention = self.softmax(attention / (self.scale ** (1 / 2))) @ v
        return attention

    def _get_spectrum(self):
        _, sq, _ = torch.svd(self.q.weight)
        _, sk, _ = torch.svd(self.k.weight)
        _, sv, _ = torch.svd(self.v.weight)
        return sq, sk, sv

    def _weight_spectral_rescale(self):
        sq, sk, sv = self._get_spectrum()
        self.q.weight = nn.Parameter(self.init_spectrum[0] / max(sq) * self.q.weight)
        self.k.weight = nn.Parameter(self.init_spectrum[1] / max(sk) * self.k.weight)
        self.v.weight = nn.Parameter(self.init_spectrum[2] / max(sv) * self.v.weight)

    def _l2att(self, q, k):
        return torch.cdist(q, k, p=2)

    def _l1att(self, q, k):
        return torch.einsum("... i d, ... j d -> ... i j", q, k)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        transformer_parameters: TransformerParameters,
        output_size: int,
        head_dimension: int,
    ):
        super().__init__()

        self.output_dimension = transformer_parameters.number_of_heads * head_dimension
        self.output_features = output_size

        self.attention_heads = nn.ModuleList(
            [
                Attention(
                    transformer_parameters=transformer_parameters,
                    output_features=head_dimension,
                    scale=self.output_dimension,
                )
                for _ in range(transformer_parameters.number_of_heads)
            ]
        )
        self.output_linear = nn.Linear(self.output_dimension, self.output_features)

    def forward(self, x):
        attentions = []
        for attention_head in self.attention_heads:
            attentions.append(attention_head(x))
        attentions = torch.cat(attentions, dim=-1)
        output = self.output_linear(attentions)
        return output
