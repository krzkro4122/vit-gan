import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from rich.progress import track
from torchvision.utils import make_grid
from cli import get_parser

import models
import utils


def exp_mov_avg(Gs, G, alpha=0.999, global_step=999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def train():
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, args.latent_dim))).to(
        device
    )
    for step in track(range(args.steps + 1)):
        # Train Discriminator
        discriminator_optimizer.zero_grad()

        # Forward + Backward with real images
        real_image = next(data_loader).to(device)
        real_label = torch.ones(args.batch_size).to(device)
        real_logit = discriminator(real_image).flatten()
        loss_discriminator_real = criterion(real_logit, real_label)
        loss_discriminator_real.backward()

        # Forward + Backward with fake images
        latent_vector = torch.FloatTensor(
            np.random.normal(0, 1, (args.batch_size, args.latent_dim))
        ).to(device)
        fake_image = generator(latent_vector)
        fake_label = torch.zeros(args.batch_size).to(device)
        fake_logit = discriminator(fake_image).flatten()
        loss_discriminator_fake = criterion(fake_logit, fake_label)
        loss_discriminator_fake.backward()

        discriminator_optimizer.step()

        # Train Generator
        generator_optimizer.zero_grad()
        fake_image = generator(
            torch.FloatTensor(
                np.random.normal(0, 1, (args.batch_size, args.latent_dim))
            ).to(device)
        )
        real_label = torch.ones(args.batch_size).to(device)
        fake_logit = discriminator(fake_image).flatten()
        loss_generator = criterion(fake_logit, real_label)
        loss_generator.backward()
        generator_optimizer.step()

        exp_mov_avg(generator_s, generator, global_step=step)

        if step % args.sample_interval == 0:
            generator.eval()
            vis = generator(fixed_noise).detach().cpu()
            vis = make_grid(vis, nrow=4, padding=5, normalize=True)
            vis = T.ToPILImage()(vis)
            vis.save(f"{output_folder_name}/samples/vis{step:05d}.jpg")
            generator.train()
            print(f"Save sample to {output_folder_name}/samples/vis{step:05d}.jpg")

        if (step + 1) % args.sample_interval == 0 or step == 0:
            # Save the checkpoints.
            torch.save(
                generator.state_dict(), f"{output_folder_name}/weights/Generator.pth"
            )
            torch.save(
                generator_s.state_dict(),
                f"{output_folder_name}/weights/Generator_ema.pth",
            )
            torch.save(
                discriminator.state_dict(),
                f"{output_folder_name}/weights/Discriminator.pth",
            )
            print("Saved model state.")


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    data_loader = utils.get_dataloader(args.data_dir, batch_size=args.batch_size)

    # Create the log folder
    output_folder_name = f"output/train_{utils.get_date_code()}"
    os.makedirs(output_folder_name, exist_ok=True)
    os.makedirs(f"{output_folder_name}/weights", exist_ok=True)
    os.makedirs(f"{output_folder_name}/samples", exist_ok=True)

    # Initialize Generator and Discriminator
    generator = models.Generator().to(device)
    generator_s = copy.deepcopy(generator)
    discriminator = models.Discriminator().to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer and lr_scheduler
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )

    # Start Training
    train()
