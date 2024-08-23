import datetime
import os
import traceback
import math
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import src.v2.modules as modules

from typing import Union
from torch import Tensor
from torch.amp.autocast_mode import autocast
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from src.v2.utils import (
    CHECKPOINT_DIR,
    log,
    convert_to_uint8,
    construct_directories,
    SAVE_DIR,
    START_TIME,
    IMAGES_DIR,
)


def diversity_loss(fake_images):
    # Get batch size and flatten images
    batch_size = fake_images.size(0)
    fake_images_flat = fake_images.view(batch_size, -1)

    # Calculate the pairwise absolute differences
    diffs = torch.cdist(fake_images_flat, fake_images_flat, p=1)

    # Sum all pairwise differences and normalize
    _diversity_loss = diffs.sum() / (batch_size * (batch_size - 1))

    return _diversity_loss


def run():
    construct_directories()

    if os.getenv("DEV", "0"):
        # Production Hyperparameters
        img_size = 32
        patch_size = 4
        in_chans = 3
        embed_dim = 512
        no_of_transformer_blocks = 3
        num_heads = 4
        mlp_ratio = 4.0
        dropout_rate = 0.3
        batch_size = 2048
        epochs = 1000
        generator_learning_rate = 3e-5
        discriminator_learning_rate = 1e-5
        discriminator_loss_threshold = 0.3
        optimizer_betas = (0.5, 0.999)
        noise_shape = in_chans, img_size, img_size
    else:
        # Development Hyperparameters
        img_size = 32
        patch_size = 4
        in_chans = 3
        embed_dim = 64
        no_of_transformer_blocks = 12
        num_heads = 8
        mlp_ratio = 4.0
        dropout_rate = 0.1
        batch_size = 64
        epochs = 100
        generator_learning_rate = 3e-5
        discriminator_learning_rate = 1e-5
        discriminator_loss_threshold = 0.3
        optimizer_betas = (0.5, 0.999)
        noise_shape = in_chans, img_size, img_size

    disc_losses = np.array([])
    gen_losses = np.array([])
    fid_scores = np.array([])

    def construct_noise():
        return torch.randn(batch_size, *noise_shape, device=device)

    def denormalize(imgs):
        return imgs * 0.5 + 0.5  # Convert from [-1,1] to [0,1]

    def save_images(save_path: str, images: torch.Tensor):
        vutils.save_image(
            images, save_path, nrow=math.floor(math.sqrt(batch_size)), normalize=True
        )

    def save_samples(label: Union[str, int], noise: torch.Tensor):
        with torch.no_grad():
            image_samples = vit_gan.generator(noise).detach().cpu()
            image_samples = denormalize(image_samples)
            save_path = os.path.join(IMAGES_DIR, f"samples_epoch_{label}.png")
            save_images(save_path, image_samples)
        log(f"[{label=}] Saved samples to {SAVE_DIR}")

    # Data loaders
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),  # Add random horizontal flip
            transforms.RandomRotation(15),  # Add random rotation
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),  # Add color jitter
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
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

    # Scheduler - Decay learning rate by 0.1 every 100 epochs
    gen_scheduler = StepLR(gen_optimizer, step_size=100, gamma=0.1)
    disc_scheduler = StepLR(disc_optimizer, step_size=100, gamma=0.1)

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
            f"  {discriminator_loss_threshold=} "
        )
        # Training Loop
        for epoch in range(epochs):
            noise = construct_noise()
            save_samples(label=epoch, noise=noise)
            for i, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(device)

                # Train Discriminator with WGAN-GP loss
                real_output = vit_gan.discriminator(real_images)
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
                np.append(disc_losses, disc_loss.item())

                if disc_loss.item() < discriminator_loss_threshold:
                    iterations = 5
                else:
                    iterations = 1

                for _ in range(iterations):
                    # Train Generator
                    vit_gan.generator.zero_grad()
                    noise = construct_noise()
                    fake_images = vit_gan.generator(noise)
                    output = vit_gan.discriminator(fake_images)

                    gen_loss = F.binary_cross_entropy_with_logits(
                        output, torch.ones_like(output)
                    )
                    div_loss = diversity_loss(fake_images)
                    total_gen_loss = (
                        gen_loss + 0.1 * div_loss
                    )  # Weight for diversity loss

                    total_gen_loss.backward()
                    gen_optimizer.step()
                    np.append(gen_losses, gen_loss.item())

                gen_scheduler.step()
                disc_scheduler.step()

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
                        np.append(fid_scores, fid_score)
                        disc_loss_value = disc_loss.item()
                        if disc_loss_value < discriminator_loss_threshold:
                            raise Exception(
                                f"The disc loss got really small! ({disc_loss_value}) Stopping training"
                            )
                        log(
                            f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}] | "
                            f"Disc Loss: {disc_loss_value:.8f}, Gen Loss: {gen_loss.item():.4f} | "
                            f"FID: {fid_score:.4f}"
                        )
                if os.getenv("DEV", "1") == "1" and i % 50 == 0:
                    log(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}]")
                if (i + 1) % 1000 == 0:  # Save every 1000 steps
                    torch.save(
                        {
                            "epoch": epoch,
                            "generator_state_dict": vit_gan.generator.state_dict(),
                            "discriminator_state_dict": vit_gan.discriminator.state_dict(),
                            "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                            "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                            "loss": gen_loss,
                        },
                        os.path.join(
                            CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}_step_{i}.pth"
                        ),
                    )
    except KeyboardInterrupt as ke:
        log(f"{ke} raised!")
    except Exception as e:
        log(f"{e} raised!\n{traceback.format_exc()}")
    finally:
        if gen_losses and disc_losses:
            # Plotting the Generator and Discriminator Loss
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(gen_losses, label="G Loss")
            plt.plot(disc_losses, label="D Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(SAVE_DIR, "losses.png")
            )  # Save the plot as an image
            plt.close()  # Close the plot to prevent it from displaying
        if fid_scores:
            # Plotting the FID Score
            plt.figure(figsize=(10, 5))
            plt.title("FID Score During Training")
            plt.plot(fid_scores, label="FID Score")
            plt.xlabel("Iterations")
            plt.ylabel("FID")
            plt.legend()
            plt.savefig(
                os.path.join(SAVE_DIR, "fid_score.png")
            )  # Save the plot as an image
            plt.close()  # Close the plot to prevent it from displaying

        model_name = "model.ckpt"
        model_path = os.path.join(SAVE_DIR, model_name)
        log(
            f"Run took {str(datetime.datetime.now() - START_TIME)}. Saving the model to: {model_path}"
        )
        torch.save(vit_gan.state_dict(), model_path)
        noise = construct_noise()
        save_samples(label=epoch, noise=noise)
