import datetime
import os
import traceback
import math
import torch
import torchvision.utils as vutils
import torch.nn.utils as utils
import src.v2.modules as modules
from ray import tune
from typing import Any, Union
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.v2.utils import (
    CHECKPOINT_DIR,
    Config,
    diversity_loss,
    evaluate_fid,
    get_data_loader,
    gradient_penalty,
    log,
    construct_directories,
    SAVE_DIR,
    START_TIME,
    IMAGES_DIR,
    save_figures,
)


def train_model(config: dict[str, Any]):
    def construct_noise(requested_batch_size=0):
        return torch.randn(
            c.batch_size if not requested_batch_size else requested_batch_size,
            c.in_chans,
            c.img_size,
            c.img_size,
            device=device,
        )

    def denormalize(imgs):
        return imgs * 0.5 + 0.5

    def save_images(save_path: str, images: torch.Tensor):
        vutils.save_image(
            images, save_path, nrow=math.floor(math.sqrt(c.batch_size)), normalize=True
        )

    def save_samples(label: Union[str, int], noise: torch.Tensor):
        with torch.no_grad():
            image_samples = vit_gan.generator(noise).detach().cpu()
            image_samples = denormalize(image_samples)
            save_path = os.path.join(IMAGES_DIR, f"samples_epoch_{label}.png")
            save_images(save_path, image_samples)
        log(f"[{label=}] Saved samples to {SAVE_DIR}")

    def train_generator():
        gen_optimizer.zero_grad()
        fake_images = vit_gan.generator(construct_noise())
        output = vit_gan.discriminator(fake_images).view(-1)

        gen_loss = -torch.mean(output)
        div_loss = diversity_loss(fake_images)
        total_gen_loss = gen_loss + 0.1 * div_loss

        total_gen_loss.backward()
        gen_optimizer.step()

        gen_grad_norm = utils.clip_grad_norm_(
            vit_gan.generator.parameters(), max_norm=0.5
        )
        gen_losses.append(gen_loss.cpu().item())
        gradient_norms_gen.append(gen_grad_norm.cpu().item())

    def train_discriminator():
        noise_level = 0.1
        noisy_real_images = real_images + noise_level * torch.randn_like(real_images)
        noisy_fake_images = vit_gan.generator(
            construct_noise()
        ) + noise_level * torch.randn_like(real_images)

        disc_optimizer.zero_grad()
        real_output = vit_gan.discriminator(noisy_real_images).view(-1)
        fake_output = vit_gan.discriminator(noisy_fake_images.detach()).view(-1)

        disc_loss = -(torch.mean(real_output) - torch.mean(fake_output))

        gp = gradient_penalty(
            vit_gan.discriminator, noisy_real_images, noisy_fake_images, device
        )
        disc_loss += c.lambda_gp * gp

        disc_loss.backward()
        disc_optimizer.step()

        utils.clip_grad_norm_(vit_gan.discriminator.parameters(), max_norm=5.0)

        disc_loss_ma.update(disc_loss.item())
        disc_losses.append(disc_loss.cpu().item())

        disc_real_acc = (real_output > 0).float().mean().cpu().item()
        disc_fake_acc = (fake_output < 0).float().mean().cpu().item()
        disc_real_accuracies.append(disc_real_acc)
        disc_fake_accuracies.append(disc_fake_acc)

        disc_grad_norm = utils.clip_grad_norm_(
            vit_gan.discriminator.parameters(), max_norm=5.0
        )
        gradient_norms_disc.append(disc_grad_norm.cpu().item())

    torch.cuda.empty_cache()
    construct_directories()

    if config:
        c = Config(**config)

    best_fid_score = float("inf")
    disc_losses = []
    gen_losses = []
    fid_scores = []
    gradient_norms_gen = []
    gradient_norms_disc = []
    disc_real_accuracies = []
    disc_fake_accuracies = []

    # Data loaders
    data_loader = get_data_loader(c)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_gan = modules.HybridViTGAN(
        img_size=c.img_size,
        patch_size=c.patch_size,
        embed_dim=c.embed_dim,
        depth=c.no_of_transformer_blocks,
        num_heads=c.num_heads,
        mlp_ratio=c.mlp_ratio,
        in_chans=c.in_chans,
    ).to(device)

    vit_gan = modules.load_pretrained_discriminator(vit_gan)
    vit_gan.train()

    gen_optimizer = Adam(
        vit_gan.generator.parameters(),
        lr=c.generator_learning_rate,
        betas=(c.optimizer_beta1, c.optimizer_beta2),
        weight_decay=c.gen_weight_decay,
    )
    disc_optimizer = Adam(
        vit_gan.discriminator.parameters(),
        lr=c.discriminator_learning_rate,
        betas=(c.optimizer_beta1, c.optimizer_beta2),
        weight_decay=c.disc_weight_decay,
    )

    gen_scheduler = ReduceLROnPlateau(
        gen_optimizer, mode="min", factor=0.5, patience=5, threshold=0.01, min_lr=1e-6
    )
    disc_scheduler = ReduceLROnPlateau(
        disc_optimizer, mode="min", factor=0.5, patience=5, threshold=0.01, min_lr=1e-6
    )

    disc_loss_ma = modules.MovingAverage(alpha=0.9)
    early_stopping = modules.EarlyStopping(patience=20, min_delta=10.0)

    try:
        log(f"Starting training at: {str(datetime.datetime.now())}")
        log("Parameters:\n" f"{str(c)}")

        for epoch in range(c.epochs):
            noise = construct_noise()
            save_samples(label=epoch, noise=noise)

            for i, (real_images, _) in enumerate(data_loader):
                real_images = real_images.to(device)

                train_discriminator()

                if i % 5 == 0:
                    train_generator()

            if epoch % 5 == 0:
                fid_score = evaluate_fid(vit_gan, data_loader, device, fid_scores)

                gen_scheduler.step(fid_score)
                disc_scheduler.step(fid_score)

                if fid_score < best_fid_score:
                    best_fid_score = fid_score
                    torch.save(
                        vit_gan.state_dict(),
                        os.path.join(
                            CHECKPOINT_DIR,
                            f"best_model_epoch_{epoch}_fid_{int(fid_score)}.pth",
                        ),
                    )

                tune.report(fid_score=fid_score)

                if early_stopping.should_stop(fid_score):
                    log(
                        f"Early stopping triggered at epoch {epoch} with FID: {fid_score}"
                    )
                    break

            save_figures(
                disc_losses=disc_losses,
                gen_losses=gen_losses,
                fid_scores=fid_scores,
                gradient_norms_gen=gradient_norms_gen,
                gradient_norms_disc=gradient_norms_disc,
                disc_real_accuracies=disc_real_accuracies,
                disc_fake_accuracies=disc_fake_accuracies,
            )
    except KeyboardInterrupt as ke:
        log(f"{ke} raised!")
    except Exception as e:
        log(f"Exception: {e}\n{traceback.format_exc()}")
    finally:
        save_figures(
            disc_losses=disc_losses,
            gen_losses=gen_losses,
            fid_scores=fid_scores,
            gradient_norms_gen=gradient_norms_gen,
            gradient_norms_disc=gradient_norms_disc,
            disc_real_accuracies=disc_real_accuracies,
            disc_fake_accuracies=disc_fake_accuracies,
        )
        model_path = os.path.join(SAVE_DIR, "final_model.ckpt")
        torch.save(vit_gan.state_dict(), model_path)
        noise = construct_noise()
        save_samples(label=epoch, noise=noise)
        log(
            f"Run took {str(datetime.datetime.now() - START_TIME)}. Saving the model to: {model_path}"
        )
