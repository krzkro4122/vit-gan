import datetime
import os
import traceback
import math
import torch
from ray import tune
import ray
import torchvision.utils as vutils
import torch.nn.utils as utils
import src.v2.modules as modules
from ray import tune
from typing import Any, Optional, Union
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.v2.utils import (
    CHECKPOINT_DIR,
    NOISE_DIR,
    Config,
    diversity_loss,
    evaluate_fid,
    get_data_loader,
    gradient_penalty,
    log,
    construct_directories,
    SAVE_DIR,
    START_TIME,
    INPUT_DIR,
    IMAGES_DIR,
    save_figures,
)


def train_model(config: Optional[dict[str, Any]] = None):
    def construct_noise():
        return torch.randn(
            c.batch_size,
            c.input_channels,
            c.image_size,
            c.image_size,
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

    def save_noise(label: Union[str, int], noise: torch.Tensor):
        save_path = os.path.join(NOISE_DIR, f"noise_epoch_{label}.png")
        save_images(save_path, noise)

    def save_input(label: Union[str, int], images: torch.Tensor):
        save_path = os.path.join(INPUT_DIR, f"input_epoch_{label}.png")
        save_images(save_path, images)

    def train_generator():
        gen_optimizer.zero_grad()
        fake_images = vit_gan.generator(construct_noise())
        output = vit_gan.discriminator(fake_images).view(-1)

        loss = -torch.mean(output)
        div_loss = diversity_loss(fake_images)
        total_gen_loss = loss + 0.1 * div_loss

        total_gen_loss.backward()
        utils.clip_grad_norm_(vit_gan.generator.parameters(), max_norm=0.5)
        gen_optimizer.step()

        gen_losses.append(loss.item())
        grad_norm = sum(
            p.grad.detach().norm().item() for p in vit_gan.generator.parameters()
        )
        gradient_norms_gen.append(grad_norm)

        return loss, grad_norm

    def train_on_real_data():
        noise_level = 0.1
        noisy_real_images = real_images + noise_level * torch.randn_like(real_images)
        noise = construct_noise()

        noisy_fake_images = vit_gan.generator(noise).detach().to(
            device
        ) + noise_level * torch.randn_like(real_images)

        disc_optimizer.zero_grad()
        real_output = vit_gan.discriminator(noisy_real_images).view(-1)
        fake_output = vit_gan.discriminator(noisy_fake_images).view(-1)

        loss = -(torch.mean(real_output) - torch.mean(fake_output))

        gp = gradient_penalty(
            vit_gan.discriminator, noisy_real_images, noisy_fake_images, device
        )
        loss += c.lambda_gp * gp

        loss.backward()
        utils.clip_grad_norm_(vit_gan.discriminator.parameters(), max_norm=5.0)
        disc_optimizer.step()

        disc_loss_ma.update(loss.item())
        disc_losses.append(loss.item())

        real_accuracy = (real_output > 0).float().mean().item()
        fake_accuracy = (fake_output < 0).float().mean().item()
        disc_real_accuracies.append(real_accuracy)
        disc_fake_accuracies.append(fake_accuracy)

        grad_norm = sum(
            p.grad.detach().norm().item() for p in vit_gan.discriminator.parameters()
        )
        gradient_norms_disc.append(grad_norm)

        return loss, grad_norm, fake_accuracy, real_accuracy

    torch.cuda.empty_cache()
    construct_directories()

    c = Config() if not config else Config(**config)

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

    vit_gan = modules.CNNGAN(c).to(device)

    vit_gan = modules.load_pretrained_discriminator(vit_gan)
    vit_gan.train()

    gen_optimizer = AdamW(
        vit_gan.generator.parameters(),
        lr=c.generator_learning_rate,
        betas=(c.optimizer_beta1, c.optimizer_beta2),
        weight_decay=c.generator_weight_decay,
    )
    disc_optimizer = AdamW(
        vit_gan.discriminator.parameters(),
        lr=c.discriminator_learning_rate,
        betas=(c.optimizer_beta1, c.optimizer_beta2),
        weight_decay=c.discriminator_weight_decay,
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
            save_noise(label=epoch, noise=noise)
            save_samples(label=epoch, noise=noise)

            for i, (real_images, _) in enumerate(data_loader):
                if i == 0:
                    save_input(label=epoch, images=real_images)

                real_images = real_images.to(device)

                disc_loss, disc_grad_norm, disc_fake_acc, disc_real_acc = (
                    train_on_real_data()
                )

                if i % c.generator_skips == 0:
                    gen_loss, gen_grad_norm = train_generator()

            fid_score = evaluate_fid(vit_gan, data_loader, device, fid_scores)

            # gen_scheduler.step(fid_score)
            # disc_scheduler.step(fid_score)

            if fid_score < best_fid_score:
                best_fid_score = fid_score
                torch.save(
                    vit_gan.state_dict(),
                    os.path.join(
                        CHECKPOINT_DIR,
                        f"best_model_epoch_{epoch}_fid_{int(fid_score)}.pth",
                    ),
                )
            log(
                f"Epoch [{epoch}/{c.epochs}] | Disc Loss: {disc_loss.item():.8f}, Gen Loss: {gen_loss.item():.4f} | FID: {fid_score:.4f} | Disc Real Acc: {disc_real_acc:.4f} | Disc Fake Acc: {disc_fake_acc:.4f} | Grad Norm Gen: {gen_grad_norm:.4f} | Grad Norm Disc: {disc_grad_norm:.4f}"
            )

            if config:
                tune.report(fid_score=fid_score)

            if early_stopping.should_stop(fid_score):
                log(f"Early stopping triggered at epoch {epoch} with FID: {fid_score}")
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


def train_with_ray():
    os.environ["RAY_TMPDIR"] = "/tmp/ray_tmp"
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    ray.init(num_gpus=1)

    search_space = {
        "generator_learning_rate": tune.loguniform(1e-6, 1e-4),
        "discriminator_learning_rate": tune.loguniform(1e-6, 1e-4),
        "embed_dim": tune.choice([128, 256, 512]),
        "num_heads": tune.choice([4, 8]),
        "batch_size": tune.choice([128, 256]),
    }

    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": 6, "gpu": 1},
        config=search_space,
        num_samples=10,
        metric="fid_score",
        mode="min",
    )

    print("Best config: ", analysis.best_config)
