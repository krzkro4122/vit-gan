import datetime
import os
import traceback
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import src.v2.modules as modules

from typing import Union
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from src.v2.utils import (
    log,
    convert_to_uint8,
    construct_directories,
    SAVE_DIR,
    START_TIME,
    IMAGES_DIR,
    NOISE_DIR,
    INPUT_DIR,
)


def run():
    construct_directories()

    # Hyperparameters
    img_size = 32
    patch_size = 16
    in_chans = 3
    embed_dim = 256
    no_of_transformer_blocks = 6
    num_heads = 8
    mlp_ratio = 4.0
    dropout_rate = 0.1
    batch_size = 256
    epochs = 10_000
    generator_learning_rate = 10e-5
    discriminator_learning_rate = 3e-5
    optimizer_betas = (0.5, 0.999)
    noise_shape = in_chans, img_size, img_size

    def construct_noise():
        return torch.randn(batch_size, *noise_shape, device=device)

    def denormalize(imgs):
        return imgs * 0.5 + 0.5  # Convert from [-1,1] to [0,1]

    def save_images(save_path: str, images: torch.Tensor):
        vutils.save_image(images, save_path, nrow=8, normalize=True)

    def save_noise(label: Union[str, int]):
        noise = construct_noise()
        save_path = os.path.join(NOISE_DIR, f"noise_epoch_{label}.png")
        save_images(save_path, noise)
        log(f"[{label=}] Saved noise to {SAVE_DIR}")
        return noise

    def save_samples(label: Union[str, int], noise: torch.Tensor):
        with torch.no_grad():
            image_samples = vit_gan.generator(noise).detach().cpu()
            image_samples = denormalize(image_samples)
            save_path = os.path.join(IMAGES_DIR, f"samples_epoch_{label}.png")
            save_images(save_path, image_samples)
        log(f"[{label=}] Saved samples to {SAVE_DIR}")

    def save_input(label: Union[str, int], loaded_images: torch.Tensor):
        save_path = os.path.join(INPUT_DIR, f"input_{label}.png")
        save_images(images=loaded_images, save_path=save_path)
        log(f"[{label=}] Saved input to {SAVE_DIR}")

    # Data loaders
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts to [0,1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalizes to [-1,1]
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="~/rep/me/vit-gan/data/cifar-10-python/",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize ViTGAN
    vit_gan = modules.ViTGAN(
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        no_of_transformer_blocks,
        num_heads,
        mlp_ratio,
        dropout_rate,
    ).to(device)
    gen_optimizer = Adam(
        vit_gan.generator.parameters(),
        lr=generator_learning_rate,
        betas=optimizer_betas,
    )
    disc_optimizer = Adam(
        vit_gan.discriminator.parameters(),
        lr=discriminator_learning_rate,
        betas=optimizer_betas,
    )

    fid = FrechetInceptionDistance(feature=2048).to(device)

    try:
        log(f"Starting training at: {str(datetime.datetime.now())}")
        log(
            "Parameters:\n"
            f"  {img_size=}\n"
            f"  {patch_size=}\n"
            f"  {in_chans=}\n"
            f"  {embed_dim=}\n"
            f"  {no_of_transformer_blocks=}\n"
            f"  {num_heads=}\n"
            f"  {mlp_ratio=}\n"
            f"  {dropout_rate=}\n"
            f"  {batch_size=}\n"
            f"  {epochs=}\n"
            f"  {generator_learning_rate=}\n"
            f"  {discriminator_learning_rate=}\n"
            f"  {optimizer_betas=}\n"
            f"  {noise_shape=} "
        )
        # Training Loop
        for epoch in range(epochs):
            noise = construct_noise()
            save_samples(label=epoch, noise=noise)
            for i, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(device)

                if i % 4 == 0:
                    # Train Discriminator
                    vit_gan.discriminator.zero_grad()
                    real_output = vit_gan.discriminator(real_images)
                    noise = construct_noise()
                    fake_images = vit_gan.generator(noise)
                    fake_output = vit_gan.discriminator(fake_images.detach())
                    disc_loss_real = F.binary_cross_entropy_with_logits(
                        real_output, torch.ones_like(real_output)
                    )
                    disc_loss_fake = F.binary_cross_entropy_with_logits(
                        fake_output, torch.zeros_like(fake_output)
                    )
                    disc_loss = disc_loss_real + disc_loss_fake
                    disc_loss.backward()
                    disc_optimizer.step()

                # Train Generator
                vit_gan.generator.zero_grad()
                noise = construct_noise()
                fake_images = vit_gan.generator(noise)
                output = vit_gan.discriminator(fake_images)
                gen_loss = F.binary_cross_entropy_with_logits(
                    output, torch.ones_like(output)
                )
                gen_loss.backward()
                gen_optimizer.step()

                if i % 100 == 0:
                    with torch.no_grad():
                        noise = construct_noise()
                        fake_images = vit_gan.generator(noise).detach()

                        real_images_uint8 = convert_to_uint8(real_images)
                        fake_images_uint8 = convert_to_uint8(fake_images)

                        fid.update(real_images_uint8, real=True)
                        fid.update(fake_images_uint8, real=False)

                        fid_score = fid.compute().item()
                        fid.reset()
                        disc_loss_value = disc_loss.item()
                        if disc_loss_value < 1e-7:
                            raise Exception(
                                f"The disc loss got really small! ({disc_loss_value}) Stopping training"
                            )
                        log(
                            f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}] | "
                            f"Disc Loss: {disc_loss_value:.6f}, Gen Loss: {gen_loss.item():.4f} | "
                            f"FID: {fid_score:.4f}"
                        )
    except KeyboardInterrupt as ke:
        log(f"{ke} raised!")
    except Exception as e:
        log(f"{e} raised!\n{traceback.format_exc()}")
    finally:
        model_name = "model.ckpt"
        model_path = os.path.join(SAVE_DIR, model_name)
        log(
            f"Run took {str(datetime.datetime.now() - START_TIME)}. Saving the model to: {model_path}"
        )
        torch.save(vit_gan.state_dict(), model_path)
        noise = construct_noise()
        save_samples(label=epoch, noise=noise)
