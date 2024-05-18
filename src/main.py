from constants import Constants
from rich.progress import track
from torchvision.utils import make_grid
import copy
import models
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import utils


def exp_mov_avg(Gs, G, alpha=0.999, global_step=999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def save_checkpoint(step: int):
    if (step + 1) % Constants.SAMPLE_INTERVAL == 0 or step == 0:
        torch.save(
            generator.state_dict(),
            f"{Constants.OUTPUT_FOLDER_NAME}/weights/Generator.pth",
        )
        torch.save(
            secondary_generator.state_dict(),
            f"{Constants.OUTPUT_FOLDER_NAME}/weights/Generator_ema.pth",
        )
        torch.save(
            discriminator.state_dict(),
            f"{Constants.OUTPUT_FOLDER_NAME}/weights/Discriminator.pth",
        )
        print("Saved model state.")


def save_sample(step: int | None, fixed_noise):
    if step % Constants.SAMPLE_INTERVAL == 0:
        generator.eval()
        sample = generator(fixed_noise).detach().cpu()
        sample = make_grid(sample, nrow=4, normalize=True)
        sample = T.ToPILImage()(sample)
        sample.save(f"{Constants.OUTPUT_FOLDER_NAME}/samples/vis{step:05d}.jpg")
        generator.train()
        print(
            f"Saved sample to {Constants.OUTPUT_FOLDER_NAME}/samples/vis{step:05d}.jpg"
        )


def save_noise(fixed_noise):
    generator.eval()
    sample = generator(fixed_noise).detach().cpu()
    sample = make_grid(sample, nrow=4, normalize=True)
    sample = T.ToPILImage()(sample)
    sample.save(f"{Constants.OUTPUT_FOLDER_NAME}/samples/noise.jpg")
    generator.train()
    print(f"Saved noise to {Constants.OUTPUT_FOLDER_NAME}/samples/noise.jpg")


def full_pass_on_real_images():
    # Forward + Backward
    real_image = next(data_loader).to(device)
    real_label = torch.ones(Constants.BATCH_SIZE).to(device)
    real_logit = discriminator(real_image).flatten()
    loss_discriminator_real = criterion(real_logit, real_label)
    loss_discriminator_real.backward()


def full_pass_on_fake_images():
    # Forward + Backward
    latent_vector = torch.FloatTensor(
        np.random.normal(
            0, 1, (Constants.BATCH_SIZE, Constants.LATENT_VECTOR_DIMENSIONS)
        )
    ).to(device)
    fake_image = generator(latent_vector)
    fake_label = torch.zeros(Constants.BATCH_SIZE).to(device)
    fake_logit = discriminator(fake_image).flatten()
    loss_discriminator_fake = criterion(fake_logit, fake_label)
    loss_discriminator_fake.backward()


def train_generator():
    generator_optimizer.zero_grad()
    fake_image = generator(
        torch.FloatTensor(
            np.random.normal(
                0, 1, (Constants.BATCH_SIZE, Constants.LATENT_VECTOR_DIMENSIONS)
            )
        ).to(device)
    )
    real_label = torch.ones(Constants.BATCH_SIZE).to(device)
    fake_logit = discriminator(fake_image).flatten()
    loss_generator = criterion(fake_logit, real_label)
    loss_generator.backward()
    generator_optimizer.step()


def train():
    fixed_noise = torch.FloatTensor(
        np.random.normal(0, 1, (16, Constants.LATENT_VECTOR_DIMENSIONS))
    ).to(device)
    save_noise(fixed_noise)
    for step in track(range(Constants.NUMBER_OF_STEPS + 1)):
        discriminator_optimizer.zero_grad()
        full_pass_on_real_images()
        full_pass_on_fake_images()
        discriminator_optimizer.step()
        train_generator()
        exp_mov_avg(secondary_generator, generator, global_step=step)
        save_sample(step, fixed_noise)
        save_checkpoint(step)


if __name__ == "__main__":
    device = utils.get_device()
    data_loader = utils.get_dataloader()
    utils.create_output_directories()

    generator = models.Generator().to(device)
    secondary_generator = copy.deepcopy(generator)
    discriminator = models.Discriminator().to(device)

    criterion = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=Constants.LEARNING_RATE,
        betas=(Constants.BETA1, Constants.BETA2),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=Constants.LEARNING_RATE,
        betas=(Constants.BETA1, Constants.BETA2),
    )
    train()
