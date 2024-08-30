import datetime
import os
import traceback
import math
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import torch.nn.utils as utils
import src.v2.modules as modules

from typing import Any, Optional, Union
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


# New gradient penalty function for WGAN-GP
def gradient_penalty(discriminator, real_images, fake_images, device):
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated_images.requires_grad_(True)

    interpolated_output = discriminator(interpolated_images)

    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(interpolated_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


# Existing diversity loss function
def diversity_loss(fake_images):
    batch_size = fake_images.size(0)
    fake_images_flat = fake_images.view(batch_size, -1)
    diffs = torch.cdist(fake_images_flat, fake_images_flat, p=1)
    _diversity_loss = diffs.sum() / (batch_size * (batch_size - 1))
    return _diversity_loss


# Main training loop
def run():
    torch.cuda.empty_cache()
    construct_directories()

    # Production Hyperparameters
    img_size = 32
    patch_size = 8
    in_chans = 3
    embed_dim = 64
    no_of_transformer_blocks = 8
    num_heads = 8
    mlp_ratio = 4.0
    dropout_rate = 0.05
    batch_size = 256
    epochs = 2000
    generator_learning_rate = 5e-4
    discriminator_learning_rate = 5e-4
    optimizer_betas = (0.5, 0.999)
    noise_shape = (
        batch_size,
        in_chans,
        img_size,
        img_size,
    )  # Update to reflect latent_dim
    disc_weight_decay = 1e-4
    gen_weight_decay = 0
    lambda_gp = 10  # Gradient penalty coefficient

    if os.getenv("DEV", "0") == "1":
        batch_size = 64
        epochs = 100
    noise_shape = (
        batch_size,
        in_chans,
        img_size,
        img_size,
    )

    best_fid_score = float("inf")
    disc_losses = []
    gen_losses = []
    fid_scores = []
    gradient_norms_gen = []
    gradient_norms_disc = []
    disc_real_accuracies = []
    disc_fake_accuracies = []

    def construct_noise(requested_batch_size=0):
        return torch.randn(
            batch_size if not requested_batch_size else requested_batch_size,
            in_chans,
            img_size,
            img_size,
            device=device,
        )

    def denormalize(imgs):
        return imgs * 0.5 + 0.5

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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ViTGAN
    vit_gan = modules.HybridViTGAN(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=no_of_transformer_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        in_chans=in_chans,
    ).to(device)

    vit_gan = modules.load_pretrained_discriminator(vit_gan)
    vit_gan.train()

    gen_optimizer = Adam(
        vit_gan.generator.parameters(),
        lr=generator_learning_rate,
        betas=optimizer_betas,
        weight_decay=gen_weight_decay,
    )
    disc_optimizer = Adam(
        vit_gan.discriminator.parameters(),
        lr=discriminator_learning_rate,
        betas=optimizer_betas,
        weight_decay=disc_weight_decay,
    )

    # Scheduler
    gen_scheduler = ReduceLROnPlateau(
        gen_optimizer, mode="min", factor=0.5, patience=5, threshold=0.01, min_lr=1e-6
    )
    disc_scheduler = ReduceLROnPlateau(
        disc_optimizer, mode="min", factor=0.5, patience=5, threshold=0.01, min_lr=1e-6
    )

    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Initialize moving average and early stopping
    disc_loss_ma = MovingAverage(alpha=0.9)
    early_stopping = EarlyStopping(patience=20, min_delta=25.0)

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
            f"  {noise_shape=}\n"
            f"  {gen_weight_decay=}\n"
            f"  {disc_weight_decay=}\n"
            f"  {lambda_gp=}"
        )
        for epoch in range(epochs):
            noise = construct_noise()
            save_samples(label=epoch, noise=noise)
            for i, (real_images, _) in enumerate(train_loader):
                real_images = real_images.to(device)

                # Add instance noise to discriminator's inputs
                noise_level = 0.1  # Start with small noise, adjust if needed
                noisy_real_images = real_images + noise_level * torch.randn_like(
                    real_images
                )
                noisy_fake_images = vit_gan.generator(
                    construct_noise(real_images.size(0))
                ) + noise_level * torch.randn_like(real_images)

                # Train Discriminator
                disc_optimizer.zero_grad()
                real_output = vit_gan.discriminator(noisy_real_images).view(-1)
                fake_output = vit_gan.discriminator(noisy_fake_images.detach()).view(-1)

                # WGAN-GP Loss for Discriminator
                disc_loss = -(torch.mean(real_output) - torch.mean(fake_output))

                # Compute Gradient Penalty
                gp = gradient_penalty(
                    vit_gan.discriminator, noisy_real_images, noisy_fake_images, device
                )
                disc_loss += lambda_gp * gp

                # Gradient clipping for the discriminator
                disc_loss.backward()
                disc_optimizer.step()

                utils.clip_grad_norm_(vit_gan.discriminator.parameters(), max_norm=5.0)

                # Update moving average of discriminator loss
                disc_loss_ma.update(disc_loss.item())
                disc_losses.append(disc_loss.item())

                # Log discriminator accuracy
                disc_real_acc = (real_output > 0).float().mean().item()
                disc_fake_acc = (fake_output < 0).float().mean().item()
                disc_real_accuracies.append(disc_real_acc)
                disc_fake_accuracies.append(disc_fake_acc)

                # Log gradient norm for discriminator
                disc_grad_norm = utils.clip_grad_norm_(
                    vit_gan.discriminator.parameters(), max_norm=5.0
                )
                gradient_norms_disc.append(disc_grad_norm)

                # Train Generator (Only once for every 5 discriminator updates)
                if i % 5 == 0:
                    gen_optimizer.zero_grad()
                    fake_images = vit_gan.generator(construct_noise())
                    output = vit_gan.discriminator(fake_images).view(-1)

                    # WGAN-GP Loss for Generator
                    gen_loss = -torch.mean(output)

                    div_loss = diversity_loss(fake_images)
                    total_gen_loss = gen_loss + 0.1 * div_loss

                    total_gen_loss.backward()

                    # Gradient clipping for the generator
                    gen_optimizer.step()

                    gen_grad_norm = utils.clip_grad_norm_(
                        vit_gan.generator.parameters(), max_norm=0.5
                    )
                    gen_losses.append(gen_loss.item())
                    gradient_norms_gen.append(gen_grad_norm)

                if i % (len(train_loader) // 20) == 0:
                    with torch.no_grad():
                        noise = construct_noise()
                        fake_images = vit_gan.generator(noise).detach()

                        real_images_uint8 = convert_to_uint8(real_images)
                        fake_images_uint8 = convert_to_uint8(fake_images)

                        fid.update(real_images_uint8, real=True)
                        fid.update(fake_images_uint8, real=False)

                        fid_score = fid.compute().item()
                        fid.reset()
                        fid_scores.append(fid_score)

                        # Step learning rate schedulers
                        gen_scheduler.step(fid_score)
                        disc_scheduler.step(fid_score)

                        # Check for early stopping based on FID
                        if early_stopping.should_stop(fid_score):
                            log(
                                f"Early stopping triggered at epoch {epoch}, step {i}, ({fid_score=}/{early_stopping.best_score=})"
                            )
                            break

                        # Save best model based on FID
                        if fid_score < best_fid_score:
                            best_fid_score = fid_score
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "batch": i,
                                    "vit_gan_state_dict": vit_gan.state_dict(),
                                    "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                                    "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                                    "gen_loss": gen_loss,
                                    "disc_loss": disc_loss,
                                },
                                os.path.join(
                                    CHECKPOINT_DIR,
                                    f"checkpoint_epoch_{epoch}_step_{i}_best_fid.pth",
                                ),
                            )
                        log(
                            f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}] | Disc Loss: {disc_loss.item():.8f}, Gen Loss: {gen_loss.item():.4f} | FID: {fid_score:.4f} | Disc Real Acc: {disc_real_acc:.4f} | Disc Fake Acc: {disc_fake_acc:.4f} | Grad Norm Gen: {gen_grad_norm:.4f} | Grad Norm Disc: {disc_grad_norm:.4f}"
                        )
        if epoch % 10 == 0:  # Changed to correctly save plots at intervals
            if gen_losses and disc_losses:
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(gen_losses, label="G Loss")
                plt.plot(disc_losses, label="D Loss")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(os.path.join(SAVE_DIR, "losses.png"))
                plt.close()

            if fid_scores:
                plt.figure(figsize=(10, 5))
                plt.title("FID Score During Training")
                plt.plot(fid_scores, label="FID Score")
                plt.xlabel("Iterations")
                plt.ylabel("FID")
                plt.legend()
                plt.savefig(os.path.join(SAVE_DIR, "fid_score.png"))
                plt.close()

            if gradient_norms_gen and gradient_norms_disc:
                plt.figure(figsize=(10, 5))
                plt.title("Gradient Norms During Training")
                plt.plot(gradient_norms_gen, label="Gen Grad Norm")
                plt.plot(gradient_norms_disc, label="Disc Grad Norm")
                plt.xlabel("Iterations")
                plt.ylabel("Gradient Norm")
                plt.legend()
                plt.savefig(os.path.join(SAVE_DIR, "grad_norms.png"))
                plt.close()

            if disc_real_accuracies and disc_fake_accuracies:
                plt.figure(figsize=(10, 5))
                plt.title("Discriminator Accuracy During Training")
                plt.plot(disc_real_accuracies, label="Disc Real Acc")
                plt.plot(disc_fake_accuracies, label="Disc Fake Acc")
                plt.xlabel("Iterations")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig(os.path.join(SAVE_DIR, "disc_accuracy.png"))
                plt.close()
    except KeyboardInterrupt as ke:
        log(f"{ke} raised!")
    # except Exception as e:`
    #     log(f"{e} raised!\n{traceback.format_exc()}")`
    finally:
        model_path = os.path.join(SAVE_DIR, "final_model.ckpt")
        log(
            f"Run took {str(datetime.datetime.now() - START_TIME)}. Saving the model to: {model_path}"
        )
        torch.save(vit_gan.state_dict(), model_path)
        noise = construct_noise()
        save_samples(label=epoch, noise=noise)
