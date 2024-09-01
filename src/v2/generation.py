import os

import torch
import torchvision.utils as vutils

from src.v2.modules import ViTGAN
from src.v2.utils import Config


def test():
    c = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize ViTGAN
    vit_gan = ViTGAN(
        c.image_size,
        c.patch_size,
        c.input_chanels,
        c.embeddings_dimension,
        c.transformer_blocks_count,
        c.attention_heads_count,
        c.mlp_ratio,
        c.dropout_rate,
    ).to(device)

    output_dir = "20240805-223515"
    model_dir = os.path.join("output", output_dir)
    model_path = os.path.join(model_dir, "model.ckpt")
    test_dir = os.path.join(model_dir, "test")

    test_dir = os.path.join(model_dir, "test")

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    vit_gan.load_state_dict(torch.load(model_path))

    vit_gan.eval()

    def construct_noise():
        return torch.randn(
            c.batch_size,
            c.input_chanels,
            c.image_size,
            c.image_size,
            device=device,
        )

    def save_images(model: ViTGAN):
        noise = construct_noise()
        sample_images = [
            model.generator(noise).detach().cpu()[i] for i in range(c.batch_size)
        ]
        images_save_path = os.path.join(test_dir, "generated_images.png")
        noise_save_path = os.path.join(test_dir, "noise.png")
        vutils.save_image(sample_images, images_save_path, nrow=8, normalize=True)
        vutils.save_image(noise, noise_save_path, nrow=8, normalize=True)
        print(f"{c.batch_size} Generated images and noise saved to {test_dir}")

    save_images(vit_gan)
