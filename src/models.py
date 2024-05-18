from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn


class SelfModulatedLayerNorm(nn.Module):
    def __init__(self, number_of_features):
        super(SelfModulatedLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(number_of_features)
        # self.gamma = nn.Parameter(torch.FloatTensor(1, 1, 1))
        # self.beta = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))  # .to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1))  # .to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.layer_norm(hl) + self.beta * w


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self, input_features, hidden_features=None, output_features=None, dropout=0.0
    ):
        super().__init__()
        if not hidden_features:
            hidden_features = input_features
        if not output_features:
            output_features = input_features
        self.linear_layer_1 = nn.Linear(input_features, hidden_features)
        self.activation = nn.GELU()
        self.linear_layer_2 = nn.Linear(hidden_features, output_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Implementation of multi-head self-attention layer using the "Einstein summation convention".
    """

    def __init__(
        self, dimensions, number_of_heads=4, head_dimensions=None, discriminator=False
    ):
        super(Attention, self).__init__()
        self.number_of_heads = number_of_heads
        self.head_dimensions = (
            int(dimensions / number_of_heads)
            if head_dimensions is None
            else head_dimensions
        )
        self.weights_dimensions = self.number_of_heads * self.head_dimensions
        self.to_query_key_values = nn.Linear(
            dimensions, self.weights_dimensions * 3, bias=False
        )
        self.scale_factor = dimensions**-0.5
        self.discriminator = discriminator
        self.output_weights = nn.Linear(self.weights_dimensions, dimensions, bias=True)

        if discriminator:
            _, spectrum, _ = torch.svd(self.to_query_key_values.weight)
            self.spectral_normalization = torch.max(spectrum)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            u, s, values = torch.svd(self.to_query_key_values.weight)
            self.to_query_key_values.weight = torch.nn.Parameter(
                self.to_query_key_values.weight
                * self.spectral_normalization
                / torch.max(s)
            )

        # Generate the q, k, v vectors
        query_key_values = self.to_query_key_values(x)
        queries, keys, values = tuple(
            rearrange(
                query_key_values,
                "b t (d k h) -> k b h t d",
                k=3,
                h=self.number_of_heads,
            )
        )

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attention = torch.cdist(queries, keys, p=2)
        else:
            attention = torch.einsum("... i d, ... j d -> ... i j", queries, keys)
        scaled_attention = attention * self.scale_factor
        attention_score = torch.softmax(scaled_attention, dim=-1)
        result = torch.einsum("... i j, ... j d -> ... i d", attention_score, values)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.output_weights(result)


class DiscriminatorEncoderBlock(nn.Module):
    def __init__(
        self,
        dimensions,
        number_of_heads=4,
        head_dimensions=None,
        dropout=0.0,
        mlp_ratio=4,
    ):
        super(DiscriminatorEncoderBlock, self).__init__()
        self.attention = Attention(
            dimensions, number_of_heads, head_dimensions, discriminator=True
        )
        self.dropout = nn.Dropout(dropout)

        self.layer_normalization_1 = nn.LayerNorm(dimensions)
        self.layer_normalization_2 = nn.LayerNorm(dimensions)

        self.multi_layer_perceptron = MultiLayerPerceptron(
            dimensions, dimensions * mlp_ratio, dropout=dropout
        )

    def forward(self, x):
        x1 = self.layer_normalization_1(x)
        x = x + self.dropout(self.attention(x1))
        x2 = self.layer_normalization_2(x)
        x = x + self.multi_layer_perceptron(x2)
        return x


class GeneratorEncoderBlock(nn.Module):
    def __init__(
        self,
        dimensions,
        number_of_heads=4,
        head_dimensions=None,
        dropout=0.0,
        mlp_ratio=4,
    ):
        super(GeneratorEncoderBlock, self).__init__()
        self.attention = Attention(dimensions, number_of_heads, head_dimensions)
        self.dropout = nn.Dropout(dropout)

        self.layer_normalization_1 = SelfModulatedLayerNorm(dimensions)
        self.layer_normalization_2 = SelfModulatedLayerNorm(dimensions)

        self.multi_layer_perceptron = MultiLayerPerceptron(
            dimensions, dimensions * mlp_ratio, dropout=dropout
        )

    def forward(self, hl, x):
        hl_temp = self.dropout(self.attention(self.layer_normalization_1(hl, x))) + hl
        hl_final = (
            self.multi_layer_perceptron(self.layer_normalization_2(hl_temp, x))
            + hl_temp
        )
        return x, hl_final


class GeneratorTransformerEncoder(nn.Module):
    def __init__(
        self,
        dimensions,
        number_of_blocks=6,
        number_of_heads=8,
        head_dimensions=None,
        dropout=0,
    ):
        super(GeneratorTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(
            dimensions, number_of_blocks, number_of_heads, head_dimensions, dropout
        )

    def _make_layers(self, dim, blocks=6, num_heads=8, dim_head=None, dropout=0):
        layers = []
        for _ in range(blocks):
            layers.append(GeneratorEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, hl, x):
        for block in self.blocks:
            x, hl = block(hl, x)
        return x, hl


class DiscriminatorTransformerEncoder(nn.Module):
    def __init__(
        self,
        dimensions,
        number_of_blocks=6,
        number_of_heads=8,
        head_dimensions=None,
        dropout=0,
    ):
        super(DiscriminatorTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(
            dimensions, number_of_blocks, number_of_heads, head_dimensions, dropout
        )

    def _make_layers(self, dim, blocks=6, num_heads=8, dim_head=None, dropout=0):
        layers = []
        for _ in range(blocks):
            layers.append(DiscriminatorEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(
        self, input_features, output_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.input_features = input_features
        self.linear = nn.Linear(input_features, output_features, bias=bias)

        self.init_weights()

    def init_weights(self):
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

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Generator(nn.Module):
    def __init__(
        self,
        initialize_size=8,
        dimensions=384,
        number_of_blocks=6,
        number_of_heads=6,
        head_dimensions=None,
        dropout=0,
        output_channels=3,
    ):
        super(Generator, self).__init__()
        self.initialize_size = initialize_size
        self.dimensions = dimensions
        self.number_of_blocks = number_of_blocks
        self.number_of_heads = number_of_heads
        self.head_dimensions = head_dimensions
        self.dropout = dropout
        self.output_channels = output_channels

        self.positional_embeddings1D = nn.Parameter(
            torch.randn(self.initialize_size * 8, dimensions)
        )

        self.multi_layer_perceptron = nn.Linear(
            1024, (self.initialize_size * 8) * self.dimensions
        )
        self.transformer_encoder = GeneratorTransformerEncoder(
            dimensions, number_of_blocks, number_of_heads, head_dimensions, dropout
        )

        # Implicit Neural Representation
        self.output_weights = nn.Sequential(
            SineLayer(dimensions, dimensions * 2, is_first=True, omega_0=30.0),
            SineLayer(
                dimensions * 2,
                self.initialize_size * 8 * self.output_channels,
                is_first=False,
                omega_0=30,
            ),
        )
        self.self_modulated_layer_norm = SelfModulatedLayerNorm(self.dimensions)

    def forward(self, noise):
        x = self.multi_layer_perceptron(noise).view(
            -1, self.initialize_size * 8, self.dimensions
        )
        x, hl = self.transformer_encoder(self.positional_embeddings1D, x)
        x = self.self_modulated_layer_norm(hl, x)
        x = self.output_weights(x)  # Replace to siren
        result = x.view(
            x.shape[0], 3, self.initialize_size * 8, self.initialize_size * 8
        )
        return result


class Discriminator(nn.Module):
    def __init__(
        self,
        input_channels=3,
        patch_size=8,
        extend_size=2,
        dimensions=384,
        number_of_blocks=6,
        number_of_heads=6,
        head_dimensions=None,
        dropout=0,
    ):
        super(Discriminator, self).__init__()
        self.patch_size = patch_size + 2 * extend_size
        self.token_dimensions = input_channels * (self.patch_size**2)
        self.project_patches = nn.Linear(self.token_dimensions, dimensions)

        self.embeddings_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dimensions))
        self.positional_embeddings1D = nn.Parameter(
            torch.randn(self.token_dimensions + 1, dimensions)
        )
        self.multi_layer_perceptron_head = nn.Sequential(
            nn.LayerNorm(dimensions), nn.Linear(dimensions, 1)
        )

        self.transformer_encoder = DiscriminatorTransformerEncoder(
            dimensions, number_of_blocks, number_of_heads, head_dimensions, dropout
        )

    def forward(self, img):
        # Generate overlapping image patches
        kernel_height = (img.shape[2] - self.patch_size) // 8 + 1
        kernel_width = (img.shape[3] - self.patch_size) // 8 + 1
        image_patches = img.unfold(2, self.patch_size, kernel_height).unfold(
            3, self.patch_size, kernel_width
        )
        image_patches = image_patches.contiguous().view(
            image_patches.shape[0],
            image_patches.shape[2] * image_patches.shape[3],
            image_patches.shape[1] * image_patches.shape[4] * image_patches.shape[5],
        )
        image_patches = self.project_patches(image_patches)
        batch_size, tokens, _ = image_patches.shape

        # Prepend the classifier token
        cls_token = repeat(self.cls_token, "() n d -> b n d", b=batch_size)
        image_patches = torch.cat((cls_token, image_patches), dim=1)

        # Plus the positional embedding
        image_patches = image_patches + self.positional_embeddings1D[: tokens + 1, :]
        image_patches = self.embeddings_dropout(image_patches)

        result = self.transformer_encoder(image_patches)
        logits = self.multi_layer_perceptron_head(result[:, 0, :])
        logits = nn.Sigmoid()(logits)
        return logits


def test_both():
    number_of_batches, dimensions = 10, 1024
    generator = Generator(initialize_size=8, dropout=0.1)
    noise = torch.FloatTensor(np.random.normal(0, 1, (number_of_batches, dimensions)))
    fake_image = generator(noise)
    discriminator = Discriminator(patch_size=8, dropout=0.1)
    discriminator_logits = discriminator(fake_image)
    print(discriminator_logits)
    print(
        f"Max: {torch.max(discriminator_logits)}, Min: {torch.min(discriminator_logits)}"
    )
