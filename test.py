import os
import torch
import src.v2.modules as modules


torch.cuda.empty_cache()

# Production Hyperparameters
img_size = 32
patch_size = 8
in_chans = 3
embed_dim = 256
no_of_transformer_blocks = 8
num_heads = 8
mlp_ratio = 4.0
dropout_rate = 0.05
batch_size = 256
epochs = 10_000
generator_learning_rate = 1e-5
discriminator_learning_rate = 1e-5
optimizer_betas = (0.5, 0.999)
noise_shape = (batch_size, 512)  # Update to reflect latent_dim
disc_weight_decay = 1e-4
gen_weight_decay = 0
lambda_gp = 10  # Gradient penalty coefficient

if os.getenv("DEV", "0") == "1":
    batch_size = 64
    epochs = 100


def construct_noise():
    return torch.randn(*noise_shape, device=device)  # Updated


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ViTGAN
vit_gan = modules.HybridViTGAN(
    img_size=img_size,
    patch_size=patch_size,
    embed_dim=embed_dim,
    depth=no_of_transformer_blocks,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
).to(device)

vit_gan = modules.load_pretrained_discriminator(vit_gan)
vit_gan.train()

test_noise = torch.randn(1, embed_dim).to(device)
generated_image = vit_gan.generator(test_noise)
print("Generated image shape:", generated_image.shape)  # Should be (1, 3, 32, 32)

discriminator_output = vit_gan.discriminator(generated_image)
print("Discriminator output shape:", discriminator_output.shape)  # Should be (1, 1)
