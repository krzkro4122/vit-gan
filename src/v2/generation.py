import os
import torch
import torchvision.utils as vutils

from src.v2.modules import ViTGAN

def test():
    # Hyperparameters
    img_size = 32
    patch_size = 4
    in_chans = 3
    embed_dim = 256
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0
    dropout_rate = 0.2
    batch_size = 64
    noise_shape = in_chans, img_size, img_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize ViTGAN
    vit_gan = ViTGAN(
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_rate,
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


    def construct_noise(batch_size: int, noise_shape: tuple):
        return torch.randn(batch_size, *noise_shape, device=device)


    def save_images(model: ViTGAN):
        noise = construct_noise(batch_size, noise_shape)
        sample_images = [
            model.generator(noise).detach().cpu()[i] for i in range(batch_size)
        ]
        images_save_path = os.path.join(test_dir, "generated_images.png")
        noise_save_path = os.path.join(test_dir, "noise.png")
        vutils.save_image(sample_images, images_save_path, nrow=8, normalize=True)
        vutils.save_image(noise, noise_save_path, nrow=8, normalize=True)
        print(f"{batch_size} Generated images and noise saved to {test_dir}")


    save_images(vit_gan)
