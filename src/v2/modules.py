import torch
import torch.nn as nn
from src.v2.utils import Config

from typing import Optional
from torchvision.models import vit_b_16, ViT_B_16_Weights


class MovingAverage:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: float):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value

    def get(self) -> float:
        return self.value if self.value is not None else 0.0


class EarlyStopping:
    def __init__(self, patience=5, min_delta=2.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def should_stop(self, current_score: float) -> bool:

        if self.best_score is None:
            self.best_score = current_score
            return False

        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


import torch
import torch.nn as nn


# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# N -> Number of Patches = IH/P * IW/P
# S -> Sequence Length   = IH/P * IW/P + 1 or N + 1 (extra 1 is of Classification Token)
# Q -> Query Sequence length (equal to S for self-attention)
# K -> Key Sequence length   (equal to S for self-attention)
# V -> Value Sequence length (equal to S for self-attention)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )  # Patch Encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, (image_size // patch_size) ** 2, embed_dim),
            requires_grad=True,
        )  # Learnable Positional Embedding
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=True
        )  # Classification Token
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(
            x
        )  # B, C, IH, IW     -> B, E, IH/P, IW/P                Split image into the patches and embed patches
        x = x.reshape(
            [B, x.shape[1], -1]
        )  # B, E, IH/P, IW/P -> B, E, (IH/P*IW/P) -> B, E, N    Flattening the patches
        x = x.permute(
            0, 2, 1
        )  # B, E, N          -> B, N, E                         Rearrange to put sequence dimension in the middle
        x = (
            x + self.pos_embedding
        )  # B, N, E          -> B, N, E                         Add positional embedding
        x = torch.cat(
            (torch.repeat_interleave(self.cls_token, B, 0), x), dim=1
        )  # B, N, E          -> B, (N+1), E       -> B, S, E    Add classification token at the start of every sequence
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads

        self.queries = nn.Linear(
            self.embed_dim, self.head_embed_dim * self.n_attention_heads
        )  # Queries projection
        self.keys = nn.Linear(
            self.embed_dim, self.head_embed_dim * self.n_attention_heads
        )  # Keys projection
        self.values = nn.Linear(
            self.embed_dim, self.head_embed_dim * self.n_attention_heads
        )  # Values projection
        self.out_projection = nn.Linear(
            self.head_embed_dim * self.n_attention_heads, self.embed_dim
        )  # Out projection

    def forward(self, x):
        b, s, e = (
            x.shape
        )  # Note: In case of self-attention Q, K and V are all equal to S

        xq = self.queries(x).reshape(
            b, s, self.n_attention_heads, self.head_embed_dim
        )  # B, Q, E      ->  B, Q, (H*HE)  ->  B, Q, H, HE
        xq = xq.permute(0, 2, 1, 3)  # B, Q, H, HE  ->  B, H, Q, HE
        xk = self.keys(x).reshape(
            b, s, self.n_attention_heads, self.head_embed_dim
        )  # B, K, E      ->  B, K, (H*HE)  ->  B, K, H, HE
        xk = xk.permute(0, 2, 1, 3)  # B, K, H, HE  ->  B, H, K, HE
        xv = self.values(x).reshape(
            b, s, self.n_attention_heads, self.head_embed_dim
        )  # B, V, E      ->  B, V, (H*HE)  ->  B, V, H, HE
        xv = xv.permute(0, 2, 1, 3)  # B, V, H, HE  ->  B, H, V, HE

        # Compute Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2)  # B, H, K, HE  ->  B, H, HE, K
        x_attention = torch.matmul(
            xq, xk
        )  # B, H, Q, HE  *   B, H, HE, K   ->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  ->  A, B, C, F   if D==E)

        x_attention /= (
            float(self.head_embed_dim) ** 0.5
        )  # Scale presoftmax values for stability

        x_attention = torch.softmax(x_attention, dim=-1)  # Compute Attention Matrix

        x = torch.matmul(
            x_attention, xv
        )  # B, H, Q, K  *  B, H, V, HE  ->  B, H, Q, HE     Compute Attention product with Values

        # Format the output
        x = x.permute(0, 2, 1, 3)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(b, s, e)  # B, Q, H, HE -> B, Q, (H*HE)

        x = self.out_projection(x)  # B, Q,(H*HE) -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))  # Skip connections
        x = x + self.dropout2(
            self.fc2(self.activation(self.fc1(self.norm2(x))))
        )  # Skip connections
        return x


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # New architectures skip fc1 and activations and directly apply fc2.
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        n_channels,
        embed_dim,
        n_layers,
        n_attention_heads,
        forward_mul,
        image_size,
        patch_size,
        n_classes,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = EmbedLayer(
            n_channels, embed_dim, image_size, patch_size, dropout=dropout
        )
        self.encoder = nn.ModuleList(
            [
                Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(
            embed_dim
        )  # Final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)

        self.apply(vit_init_weights)  # Weight initalization

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


def vit_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)


class Generator(nn.Module):
    def __init__(self, config: Config):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Input is (batch_size, input_channels, image_size, image_size)
            nn.Conv2d(
                config.input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (256, 4, 4)
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 8, 8)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 16, 16)
            nn.ConvTranspose2d(
                64,
                config.input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
            # Output is (input_channels, image_size, image_size)
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, config: Config):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input is (batch_size, input_channels, image_size, image_size)
            nn.Conv2d(
                config.input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # (512, 2, 2) - Reduced stride to avoid shrinking too much
            nn.Conv2d(512, 1, kernel_size=2, stride=1, padding=0, bias=False),
            # Output is (batch_size, 1, 1, 1)
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class ViTGenerator(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            n_channels=config.input_channels,
            embed_dim=config.embeddings_dimension,
            n_layers=config.transformer_blocks_count,
            n_attention_heads=config.attention_heads_count,
            forward_mul=config.mlp_ratio,
            image_size=config.image_size,
            patch_size=config.patch_size,
            n_classes=config.classes_count,
            dropout=config.dropout_rate,
        )
        self.linear = nn.Linear(
            config.classes_count,
            config.batch_size,
        )
        self.image_size = config.image_size
        self.input_channels = config.input_channels

    def forward(self, x):
        x = self.vit(x)
        x = self.linear(x)
        x = x.view(-1, self.input_channels, self.image_size, self.image_size)
        return x


class ViTDiscriminator(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            n_channels=config.input_channels,
            embed_dim=config.embeddings_dimension,
            n_layers=config.transformer_blocks_count,
            n_attention_heads=config.attention_heads_count,
            forward_mul=config.mlp_ratio,
            image_size=config.image_size,
            patch_size=config.patch_size,
            n_classes=config.classes_count,
            dropout=config.dropout_rate,
        )

    def forward(self, x):
        x = self.vit(x)
        return x


class ViTGAN(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.generator = ViTGenerator(config)
        self.discriminator = ViTDiscriminator(config)

    def forward(self, z):
        generated_images = self.generator(z)
        discriminator_output = self.discriminator(generated_images)
        return generated_images, discriminator_output


class CNNGAN(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def forward(self, z):
        generated_images = self.generator(z)
        discriminator_output = self.discriminator(generated_images)
        return generated_images, discriminator_output


# Load pretrained discriminator weights
def load_pretrained_discriminator(vit_gan):
    pretrained_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    if hasattr(pretrained_vit, "fc"):
        pretrained_vit.fc = nn.Linear(pretrained_vit.fc.in_features, 1)
    elif hasattr(pretrained_vit, "classifier"):
        pretrained_vit.classifier = nn.Linear(pretrained_vit.classifier.in_features, 1)
    else:
        pretrained_vit.heads.head = nn.Linear(pretrained_vit.heads.head.in_features, 1)

    vit_gan.discriminator.load_state_dict(pretrained_vit.state_dict(), strict=False)
    return vit_gan
