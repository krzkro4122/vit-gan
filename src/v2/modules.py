import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
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


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, dropout_rate=0.0
    ):
        super().__init__()
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout_rate=0.0, noise_std=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim, hidden_features=int(dim * mlp_ratio), dropout_rate=dropout_rate
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.training:  # Inject noise only during training
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x


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
        super().__init__()
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


class ViTGenerator(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            num_classes=embed_dim,
            dropout_rate=dropout_rate,
        )
        self.linear = nn.Linear(embed_dim, img_size * img_size * in_chans)
        self.img_size = img_size
        self.in_chans = in_chans

    def forward(self, x):
        x = self.vit(x)
        x = self.linear(x)
        x = x.view(-1, self.in_chans, self.img_size, self.img_size)
        return x


class ViTDiscriminator(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            num_classes=1,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = self.vit(x)
        return x


# GAN Architecture


class ViTGAN(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.generator = ViTGenerator(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            dropout_rate,
        )
        self.discriminator = ViTDiscriminator(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            dropout_rate,
        )

    def forward(self, x):
        generated_images = self.generator(x)
        discriminator_output = self.discriminator(generated_images)
        return generated_images, discriminator_output
