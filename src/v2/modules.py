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


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, input_channels, embeddings_dimension):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embeddings_dimension = embeddings_dimension
        self.patch_dim = patch_size * patch_size * input_channels
        self.proj = nn.Linear(self.patch_dim, embeddings_dimension)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, self.patch_dim)
        x = self.proj(x)
        return x


# Define an attention mechanism
class Attention(nn.Module):
    def __init__(self, dim, attention_heads_count):
        super(Attention, self).__init__()
        self.attention_heads_count = attention_heads_count
        self.head_dim = dim // attention_heads_count
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.attention_heads_count, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


# Define MLP
class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, dropout_rate=0.0
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Define the Transformer Block
class TransformerBlock(nn.Module):
    def __init__(
        self, dim, attention_heads_count, mlp_ratio=4.0, dropout_rate=0.0, noise_std=0.1
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, attention_heads_count)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim, hidden_features=int(dim * mlp_ratio), dropout_rate=dropout_rate
        )
        self.noise_std = noise_std

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.training:  # Inject noise only during training
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x


# Define Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        input_channels,
        embeddings_dimension,
        depth,
        attention_heads_count,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        num_classes=1000,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, input_channels, embeddings_dimension
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embeddings_dimension))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embeddings_dimension)
        )
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embeddings_dimension, attention_heads_count, mlp_ratio, dropout_rate
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embeddings_dimension)
        self.head = nn.Linear(embeddings_dimension, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x


# Define the HybridGenerator
class HybridGenerator(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super(HybridGenerator, self).__init__()
        self.init_size = (
            config.image_size // 4
        )  # Start with a quarter of the target size
        self.embeddings_dimension = config.embeddings_dimension
        self.linear = nn.Linear(
            config.input_channels * config.image_size * config.image_size,
            config.embeddings_dimension * self.init_size * self.init_size,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embeddings_dimension,
                    config.attention_heads_count,
                    config.mlp_ratio,
                )
                for _ in range(config.transformer_blocks_count)
            ]
        )
        self.upsample_blocks = nn.ModuleList(
            [
                UpSampleBlock(
                    config.embeddings_dimension, config.embeddings_dimension // 2
                ),
                UpSampleBlock(
                    config.embeddings_dimension // 2, config.embeddings_dimension // 4
                ),
            ]
        )
        self.final_conv = nn.Conv2d(
            config.embeddings_dimension // 4,
            config.input_channel,
            kernel_size=3,
            padding=1,
        )
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Flatten the spatial dimensions
        batch_size, channels, height, width = z.shape
        z = z.view(batch_size, -1)  # Shape: (batch_size, config.input_channels)

        # Pass through the linear layer to get the desired embedding
        x = self.linear(z).view(
            batch_size, self.embeddings_dimension, self.init_size, self.init_size
        )

        # Continue with the rest of the forward pass
        for transformer in self.transformer_blocks:
            x = (
                transformer(x.flatten(2).permute(2, 0, 1))
                .permute(1, 2, 0)
                .view(x.size())
            )
        for upsample in self.upsample_blocks:
            x = upsample(x)
        return self.tanh(self.final_conv(x))


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


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


# Define the HybridDiscriminator
class HybridDiscriminator(nn.Module):
    def __init__(self, config: Config):
        super(HybridDiscriminator, self).__init__()
        self.patch_embed = PatchEmbedding(
            config.image_size,
            config.patch_size,
            config.input_channels,
            config.embeddings_dimension,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embeddings_dimension))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, 1 + self.patch_embed.num_patches, config.embeddings_dimension
            )
        )
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embeddings_dimension,
                    config.attention_heads_count,
                    config.mlp_ratio,
                    dropout_rate=0.1,
                )
                for _ in range(config.transformer_blocks_count)
            ]
        )
        self.norm = nn.LayerNorm(config.embeddings_dimension)
        self.head = nn.Linear(config.embeddings_dimension, 1)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


class HybridViTGAN(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.generator = HybridGenerator(config)
        self.discriminator = HybridDiscriminator(config)

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
