from torch import nn

from attention import MultiHeadSelfAttention
from spectral_layer_norm import SLN
from muilti_layer_perceptron import MLP


class Transformer(nn.Module):
    def __init__(
        self,
        input_features,
        number_of_heads=4,
        attention_head_output_dimension=None,
        attention_dropout_rate=0.2,
        mlp_layers=None,
        mlp_activation="relu",
        mlp_dropout=0.2,
        spectral_rescaling=False,
        lp=2,
        **kwargs
    ):
        """
        Usual Transformer architecture using the L2-MultiheadSelfAttention module
        :param input_features: number of input features
        :param number_of_heads: number of attention head
        :param attention_head_output_dimension: output size of each attention head, default is input_features // number_of_heads
        :param attention_dropout_rate: dropout rate applied at the output of the MSA
        :param mlp_layers: list of hidden layer dimensions of the MLP module
        :param mlp_activation: activation function of the MLP module
        :param mlp_dropout: dropout applied at each MLP layer
        :param spectral_rescaling: use spectral rescaling in attention module
        """
        super(Transformer, self).__init__()

        self.input_features = input_features
        self.number_of_heads = number_of_heads
        self.head_outdim = (
            input_features // number_of_heads
            if attention_head_output_dimension is None
            else attention_head_output_dimension
        )

        self.layer_norm_1 = nn.LayerNorm(self.input_features)
        self.layer_norm_n2 = nn.LayerNorm(self.input_features)

        self.attention_dropout = nn.Dropout(attention_dropout_rate)

        self.msa = MultiHeadSelfAttention(
            self.input_features,
            self.number_of_heads,
            self.head_outdim,
            output_size=input_features,
            spectral_scaling=spectral_rescaling,
            lp=lp,
        )
        self.mlp = MLP(
            self.input_features,
            self.input_features,
            layers=mlp_layers,
            activation=mlp_activation,
            dropout_rate=mlp_dropout,
        )

    def forward(self, x):
        x1 = self.layer_norm_1(x)
        x = x + self.attention_dropout(self.msa(x1))
        x2 = self.layer_norm_n2(x)
        out = x + self.mlp(x2)
        return out


class TransformerSLN(nn.Module):
    def __init__(
        self,
        input_features,
        number_of_heads=4,
        attention_head_output_dimension=None,
        attention_dropout_rate=0.0,
        mlp_layers=None,
        mlp_activation="relu",
        mlp_dropout=0.0,
        spectral_rescaling=False,
        lp=1,
        **kwargs
    ):
        """
        Variant Transformer architecture using the L2-MultiheadSelfAttention module and SLN instead of standard LayerNorm
        :param input_features: number of input features
        :param number_of_heads: number of attention head
        :param attention_head_output_dimension: output size of each attention head, default is input_features // number_of_heads
        :param attention_dropout_rate: dropout rate applied at the output of the MSA
        :param mlp_layers: list of hidden layer dimensions of the MLP module
        :param mlp_activation: activation function of the MLP module
        :param mlp_dropout: dropout applied at each MLP layer
        :param spectral_rescaling: use spectral rescaling in attention module
        :param lp: norm used for attention, should be 1 or 2, default 2
        """
        super(TransformerSLN, self).__init__()

        self.input_features = input_features
        self.number_of_heads = number_of_heads
        self.head_outdim = (
            input_features // number_of_heads
            if attention_head_output_dimension is None
            else attention_head_output_dimension
        )

        self.layer_norm_1 = SLN(self.input_features)
        self.layer_norm_n2 = SLN(self.input_features)

        self.attention_dropout = nn.Dropout(attention_dropout_rate)

        self.msa = MultiHeadSelfAttention(
            self.input_features,
            self.number_of_heads,
            self.head_outdim,
            output_size=input_features,
            spectral_scaling=spectral_rescaling,
            lp=lp,
        )
        self.mlp = MLP(
            self.input_features,
            self.input_features,
            layers=mlp_layers,
            activation=mlp_activation,
            dropout_rate=mlp_dropout,
        )

    def forward(self, h, x):
        htmp = self.attention_dropout(self.msa(self.layer_norm_1(h, x))) + h
        hf = self.mlp(self.layer_norm_n2(htmp, x)) + htmp
        return x, hf
