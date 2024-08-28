import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.patch_dim = patch_size * patch_size * in_chans
        self.proj = nn.Linear(self.patch_dim, embed_dim)

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
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
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
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout_rate=0.0, noise_std=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
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
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout_rate=0.0,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

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
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, in_chans, num_channels=3):
        super(HybridGenerator, self).__init__()
        self.init_size = img_size // 4  # Start with a quarter of the target size
        self.embed_dim = embed_dim
        self.linear = nn.Linear(in_chans * img_size * img_size, embed_dim * self.init_size * self.init_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.upsample_blocks = nn.ModuleList([
            UpSampleBlock(embed_dim, embed_dim // 2),
            UpSampleBlock(embed_dim // 2, embed_dim // 4),
        ])
        self.final_conv = nn.Conv2d(embed_dim // 4, num_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Flatten the spatial dimensions
        batch_size, channels, height, width = z.shape
        z = z.view(batch_size, -1)  # Shape: (batch_size, in_chans * img_size * img_size)

        # Pass through the linear layer to get the desired embedding
        x = self.linear(z).view(batch_size, self.embed_dim, self.init_size, self.init_size)

        # Continue with the rest of the forward pass
        for transformer in self.transformer_blocks:
            x = transformer(x.flatten(2).permute(2, 0, 1)).permute(1, 2, 0).view(x.size())
        for upsample in self.upsample_blocks:
            x = upsample(x)
        return self.tanh(self.final_conv(x))



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
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        num_channels=3,
    ):
        super(HybridDiscriminator, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, num_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate=0.1)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

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


# Hybrid ViT-GAN
class HybridViTGAN(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        in_chans,
        num_channels=3,
    ):
        super(HybridViTGAN, self).__init__()
        self.generator = HybridGenerator(
            img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, num_channels, in_chans
        )
        self.discriminator = HybridDiscriminator(
            img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, num_channels
        )

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
