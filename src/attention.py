import torch
from torch import nn


class Attention(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        scale=None,
        spectral_scaling=False,
        lp=2,
        **kwargs,
    ):
        super(Attention, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.scale = output_features if scale is None else scale
        self.spectral_scaling = spectral_scaling
        self.lp = lp

        assert lp in [
            1,
            2,
        ], f"Unsupported norm for attention: lp={lp} but should be 1 or 2"
        self.attention_func = self._l1att if self.lp == 1 else self._l2att

        self.q = nn.Linear(self.input_features, self.output_features, bias=False)
        self.k = nn.Linear(self.input_features, self.output_features, bias=False)
        self.v = nn.Linear(self.input_features, self.output_features, bias=False)

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

        att = self.attention_func(q, k)  # we use L2 reg only in discriminator
        att = self.softmax(att / (self.scale ** (1 / 2))) @ v
        return att

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
        input_features,
        number_of_heads,
        head_dimension,
        output_size=None,
        spectral_scaling=False,
        lp=2,
        **kwargs,
    ):
        """
        Multihead self L2-attention module based on L2-attention module
        :param input_features: number of input features
        :param number_of_heads: number of attention heads
        :param head_dimension: output size of each attention head
        :param output_size: final output feature number, default is number_of_heads * head_dimension
        :param spectral_scaling: perform spectral rescaling of q, k, v for each head
        :param lp: norm used for attention, should be 1 or 2, default 2
        """
        super(MultiHeadSelfAttention, self).__init__()

        self.out_dim = number_of_heads * head_dimension
        self.output_features = self.out_dim if output_size is None else output_size

        self.attention_heads = nn.ModuleList(
            [
                Attention(
                    input_features,
                    head_dimension,
                    scale=self.out_dim,
                    spectral_scaling=spectral_scaling,
                    lp=lp,
                )
                for _ in range(number_of_heads)
            ]
        )
        self.output_linear = nn.Linear(self.out_dim, self.output_features)

    def forward(self, x):
        atts = []
        for attention_head in self.attention_heads:
            atts.append(attention_head(x))
        atts = torch.cat(atts, dim=-1)
        out = self.output_linear(atts)
        return out
