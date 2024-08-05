import datetime
import os
from random import randint
from typing import Union
import torch
import rich
import numpy as np
import torch.nn.functional as F
from torch.optim.adam import Adam
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_fid.inception import InceptionV3
import src.v2.modules as modules

START_TIME = datetime.datetime.now()
BASE_DIR = os.getenv("SCRATCH", "~")
OUTPUT_DIR = f"{BASE_DIR}/output"
SAVE_DIR = os.path.join(OUTPUT_DIR, START_TIME.strftime("%Y%m%d-%H%M%S"))
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
NOISE_DIR = os.path.join(SAVE_DIR, "noise")


def log(message: str):
    rich.print(message)
    with open(os.path.join(SAVE_DIR, "training.log"), "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


class ToTensorUInt8(object):
    def __call__(self, pic):
        # Make the array writable by copying it
        img = np.array(pic, np.uint8, copy=True)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).contiguous()  # Convert to CxHxW
        return img


def convert_to_uint8(images):
    images = (images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    return images


def calculate_inception_score(images, batch_size=32, splits=10):
    # Convert images to float32 and normalize to [0, 1]
    images = images.float() / 255.0
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(images.device)
    with torch.no_grad():
        preds = model(images)[0]
    preds = F.softmax(preds, dim=1)
    scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits : (i + 1) * len(preds) // splits, :]
        kl_div = part * (torch.log(part) - torch.log(part.mean(dim=0, keepdim=True)))
        kl_div = kl_div.sum(dim=1)
        scores.append(kl_div.mean().exp().item())
    return torch.tensor(scores).mean(), torch.tensor(scores).std()


def run():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
    if not os.path.exists(NOISE_DIR):
        os.mkdir(NOISE_DIR)

    # Hyperparameters
    img_size = 32
    patch_size = 4
    in_chans = 3
    embed_dim = 256
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0
    dropout_rate = 0.0
    batch_size = 64
    epochs = 100
    learning_rate = 2e-4
    betas = (0.5, 0.999)
    noise_shape = 1, in_chans, img_size, img_size

    def construct_noise():
        noise = []
        for i in range(batch_size):
            torch.manual_seed(i * randint(0, 1000))
            if not noise:
                new_noise = torch.randn(noise_shape, device=device)
            else:
                new_noise = torch.randn_like(noise[0], device=device)
            noise.append(new_noise)
        return torch.cat(noise, 0)

    def save_images(label: Union[str, int], model: modules.ViTGAN):
        noises = []
        sample_images = []
        for i in range(batch_size):
            noise = construct_noise().to(device)
            noises.append(noise.detach().cpu()[i])
            sample_images.append(model.generator(noise).detach().cpu()[i])

        images_save_path = os.path.join(
            IMAGES_DIR, f"generated_images_epoch_{label}.png"
        )
        noise_save_path = os.path.join(NOISE_DIR, f"noise__{label}.png")
        vutils.save_image(sample_images, images_save_path, nrow=8, normalize=True)
        vutils.save_image(noises, noise_save_path, nrow=8, normalize=True)
        log(f"[{label=}] Saved noise and sample images at epoch {label} to {SAVE_DIR}")

    # Data loaders
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            ToTensorUInt8(),
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
        depth,
        num_heads,
        mlp_ratio,
        dropout_rate,
    ).to(device)
    gen_opt = Adam(vit_gan.generator.parameters(), lr=learning_rate, betas=betas)
    disc_opt = Adam(vit_gan.discriminator.parameters(), lr=learning_rate, betas=betas)

    fid = FrechetInceptionDistance(feature=2048).to(device)

    try:
        log(f"Starting training at: {str(datetime.datetime.now())}")
        log(
            "Parameters:\n"
            f"  {img_size=}\n"
            f"  {patch_size=}\n"
            f"  {in_chans=}\n"
            f"  {embed_dim=}\n"
            f"  {depth=}\n"
            f"  {num_heads=}\n"
            f"  {mlp_ratio=}\n"
            f"  {dropout_rate=}\n"
            f"  {batch_size=}\n"
            f"  {epochs=}\n"
            f"  {learning_rate=}\n"
            f"  {betas=}\n"
            f"  {noise_shape=} "
        )
        # Training Loop
        for epoch in range(epochs):
            save_images(model=vit_gan, label=epoch)
            for i, (real_images, _) in enumerate(train_loader):
                real_images_normalized = (
                    real_images.to(device).float() / 255.0 * 2 - 1
                )  # Normalize to [-1, 1]

                # Train Discriminator
                vit_gan.discriminator.zero_grad()
                real_output = vit_gan.discriminator(real_images_normalized)
                noise = construct_noise().to(device)
                fake_images = vit_gan.generator(noise).detach()
                fake_output = vit_gan.discriminator(fake_images)
                disc_loss = F.binary_cross_entropy_with_logits(
                    real_output, torch.ones_like(real_output)
                ) + F.binary_cross_entropy_with_logits(
                    fake_output, torch.zeros_like(fake_output)
                )
                disc_loss.backward()
                disc_opt.step()

                # Train Generator
                vit_gan.generator.zero_grad()
                noise = construct_noise().to(device)
                fake_images = vit_gan.generator(noise)
                fake_output = vit_gan.discriminator(fake_images)
                gen_loss = F.binary_cross_entropy_with_logits(
                    fake_output, torch.ones_like(fake_output)
                )
                gen_loss.backward()
                gen_opt.step()

                if i % 100 == 0:
                    noise = construct_noise().to(
                        device
                    )  # Generate new noise for evaluation
                    fake_images = vit_gan.generator(noise).detach()
                    real_images_uint8 = convert_to_uint8(real_images_normalized).to(
                        device
                    )
                    fake_images_uint8 = convert_to_uint8(fake_images).to(device)
                    fid.update(real_images_uint8, real=True)
                    fid.update(fake_images_uint8, real=False)
                    is_score, is_std = calculate_inception_score(fake_images_uint8)
                    log(
                        f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}] | Disc Loss: {disc_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f} | FID: {fid.compute().item():.4f}, IS: {is_score.item():.4f} ± {is_std.item():.4f}"
                    )
                    fid.reset()

    except KeyboardInterrupt as ke:
        log(f"{ke} raised!")
    finally:
        model_name = "model.ckpt"
        model_path = os.path.join(SAVE_DIR, model_name)
        log(
            f"Run took {str(datetime.datetime.now() - START_TIME)}. Saving the model to: {model_path}"
        )
        torch.save(vit_gan.state_dict(), model_path)
        save_images("end", vit_gan)
