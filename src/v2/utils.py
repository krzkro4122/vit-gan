from matplotlib import pyplot as plt
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import datetime
import os
import rich
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


START_TIME = datetime.datetime.now()
BASE_DIR = os.getenv("SCRATCH", ".")
OUTPUT_DIR = os.path.join(f"{BASE_DIR}", "output")
SAVE_DIR = os.path.join(OUTPUT_DIR, START_TIME.strftime("%Y%m%d-%H%M%S"))
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
INPUT_DIR = os.path.join(SAVE_DIR, "input")
NOISE_DIR = os.path.join(SAVE_DIR, "noise")
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")

is_dev = int(os.getenv("DEV", "0"))


class Config(BaseModel):
    attention_heads_count: int = 4
    batch_size: int = 64
    classes_count: int = 10
    discriminator_learning_rate: float = 5e-4
    dropout_rate: float = 0.1
    embeddings_dimension: int = 128
    epochs: int = 500
    generator_learning_rate: float = 5e-4
    image_size: int = 32
    input_channels: int = 3
    mlp_ratio: int = 2
    optimizer_beta1: float = 0.5
    optimizer_beta2: float = 0.999
    patch_size: int = 4
    transformer_blocks_count: int = 6

    def __str__(self):
        return "\n".join(repr(self)[repr(self).index("(") + 1 : -1].split(", "))


def save_figures(**kwargs):
    if kwargs.get("gen_losses") and kwargs.get("disc_losses"):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        generator_iterations = [
            i * Config().generator_skips for i in range(len(kwargs["gen_losses"]))
        ]
        plt.plot(generator_iterations, kwargs["gen_losses"], label="G Loss")
        plt.plot(kwargs["disc_losses"], label="D Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, "losses.png"))
        plt.close()

    if kwargs.get("fid_scores"):
        plt.figure(figsize=(10, 5))
        plt.title("FID Score During Training")
        plt.plot(kwargs["fid_scores"], label="FID Score")
        plt.xlabel("Iterations")
        plt.ylabel("FID")
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, "fid_score.png"))
        plt.close()

    if kwargs.get("gradient_norms_gen") and kwargs.get("gradient_norms_disc"):
        plt.figure(figsize=(10, 5))
        plt.title("Gradient Norms During Training")
        generator_iterations = [
            i * Config().generator_skips for i in range(len(kwargs["gen_losses"]))
        ]
        plt.plot(
            generator_iterations, kwargs["gradient_norms_gen"], label="Gen Grad Norm"
        )
        plt.plot(kwargs["gradient_norms_disc"], label="Disc Grad Norm")
        plt.xlabel("Iterations")
        plt.ylabel("Gradient Norm")
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, "grad_norms.png"))
        plt.close()

    if kwargs.get("disc_real_accuracies") and kwargs.get("disc_fake_accuracies"):
        plt.figure(figsize=(10, 5))
        plt.title("Discriminator Accuracy During Training")
        plt.plot(kwargs["disc_real_accuracies"], label="Disc Real Acc")
        plt.plot(kwargs["disc_fake_accuracies"], label="Disc Fake Acc")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, "disc_accuracy.png"))
        plt.close()


def get_data_loader(c: Config):
    transform = transforms.Compose(
        [
            transforms.Resize(c.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * c.input_channels, [0.5] * c.input_channels
            ),  # Normalize to [-1, 1]
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="~/rep/me/vit-gan/data/cifar-10-python/",
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


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


def diversity_loss(fake_images):
    batch_size = fake_images.size(0)
    fake_images_flat = fake_images.view(batch_size, -1)
    diffs = torch.cdist(fake_images_flat, fake_images_flat, p=1)
    _diversity_loss = diffs.sum() / (batch_size * (batch_size - 1))
    return _diversity_loss


def evaluate_fid(vit_gan, data_loader, device, fid_scores):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    vit_gan.eval()

    with torch.no_grad():
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            noise = torch.randn(real_images.size(0), 3, 32, 32, device=device)
            fake_images = vit_gan.generator(noise)

            real_images_uint8 = ((real_images * 0.5 + 0.5) * 255).to(torch.uint8)
            fake_images_uint8 = ((fake_images * 0.5 + 0.5) * 255).to(torch.uint8)

            fid.update(real_images_uint8, real=True)
            fid.update(fake_images_uint8, real=False)

    fid_score = fid.compute()
    fid_score_cpu = fid_score.cpu().item()
    fid_scores.append(fid_score_cpu)
    vit_gan.train()
    return fid_score.to(device)


def construct_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(NOISE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log(message: str):
    timestamp = datetime.datetime.now().strftime("[%F %T.%f")[:-3] + "]"
    rich.print(f"{timestamp} {message}")
    with open(os.path.join(SAVE_DIR, "training.log"), "a", encoding="utf-8") as handle:
        rich.print(f"{timestamp} {message}", file=handle)


def convert_to_uint8(images):
    images = (images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    return images
