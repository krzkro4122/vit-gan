import datetime
import os
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
SAVE_DIR = os.path.join(
    f"{os.environ['SCRATCH']}/output", START_TIME.strftime("%Y%m%d-%H%M%S")
)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


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
    gen_opt = Adam(vit_gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_opt = Adam(vit_gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fid = FrechetInceptionDistance(feature=2048).to(device)

    try:
        log("Starting training!")
        # Training Loop
        for epoch in range(epochs):
            for i, (real_images, _) in enumerate(train_loader):
                real_images_normalized = (
                    real_images.to(device).float() / 255.0 * 2 - 1
                )  # Normalize to [-1, 1]

                # Train Discriminator
                vit_gan.discriminator.zero_grad()
                real_output = vit_gan.discriminator(real_images_normalized)
                noise = torch.randn(
                    real_images_normalized.size(0), in_chans, img_size, img_size
                ).to(device)
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
                fake_images = vit_gan.generator(noise)
                fake_output = vit_gan.discriminator(fake_images)
                gen_loss = F.binary_cross_entropy_with_logits(
                    fake_output, torch.ones_like(fake_output)
                )
                gen_loss.backward()
                gen_opt.step()

                if i % 100 == 0:
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

                # Save sample generated images every 10th epoch
                if (epoch + 1) % 10 == 0:
                    sample_images = vit_gan.generator(noise).detach().cpu()
                    save_path = os.path.join(
                        SAVE_DIR, f"generated_images_epoch_{epoch+1}.png"
                    )
                    vutils.save_image(sample_images, save_path, nrow=5, normalize=True)
                    log(f"Saved sample images at epoch {epoch+1} to {save_path}")

    except KeyboardInterrupt as ke:
        log(f"{ke} raised!")
    finally:
        model_name = "model.ckpt"
        model_path = os.path.join(SAVE_DIR, model_name)
        log(
            f"Run took {str(datetime.datetime.now() - START_TIME)}. Saving the model to: {model_path}"
        )
        torch.save(vit_gan.state_dict(), model_path)
