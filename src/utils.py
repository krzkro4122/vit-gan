from datetime import datetime
from typing import Optional
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.datasets as dset
from src.config import config
from src.vitgan import ViTGAN

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.utils as vutils


CONFIG_PATH_LOCAL = f"{os.environ['HOME']}/rep/vit-gan/config.json"
CONFIG_PATH_PLGRID = f"{os.environ['HOME']}/rep/code/vit-gan/config.json"


def denormalize(tensor, mean, std):
    """
    Denormalize the tensor image with mean and std.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def display_images(image_tensors: list[torch.TensorType]):
    tensors = [
        vutils.make_grid(
            image_tensor.to(config.device)[: config.image_size], normalize=True
        ).cpu()
        for image_tensor in image_tensors
    ]
    image_tensors_denormalized = torch.stack(tensors)
    cols = int(np.ceil(np.sqrt(config.batch_size)))
    rows = int(np.ceil(config.batch_size / cols))

    _, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(config.batch_size):
        axes[i].imshow(
            np.transpose(image_tensors_denormalized[i].cpu().numpy(), (1, 2, 0)),
            interpolation="nearest",
        )
        axes[i].axis("off")

    for j in range(config.batch_size, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def display_images_v2(image_tensors: list[torch.TensorType], device: str):
    tensors = [
        vutils.make_grid(
            image_tensor.to(device)[: config.image_size], normalize=True
        ).cpu()
        for image_tensor in image_tensors
    ]
    image_tensors_denormalized = torch.stack(tensors)
    image_tensors_grid = make_grid(image_tensors_denormalized)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(image_tensors_grid.cpu().numpy(), (1, 2, 0)))
    plt.show()


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Ten PC czyni CUDA!")
    else:
        print("Ten PC nie czyni CUDA'ow!")
    return device


def get_model(save_name: Optional[str] = None):
    model = ViTGAN()
    if save_name:
        checkpoint_path = os.path.join(config.ckpt_save_path, save_name)
        model.load(checkpoint_path)
        print(f"Loaded model from: {checkpoint_path}")
    return model


def save_generator_test(model):
    noise = torch.randn(
        config.image_size, config.lattent_space_size, device=config.device
    )
    fake = model.generate(noise)
    img = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(config.ckpt_save_path, "fake.png"))


def get_dataloader(train: bool, picked_dataset):
    dataset = get_dataset(train=train, picked_dataset=picked_dataset)
    return DataLoader(
        dataset,
        shuffle=True,
    )


def get_dataset(train: bool, picked_dataset):
    picked_dataset_label = picked_dataset.url.split("/")[-1].split(".")[0]
    print(f"Using dataset: {picked_dataset_label}...")
    path = os.path.abspath(f"{os.environ['SCRATCH']}/data/{picked_dataset_label}")
    print(f"Data path: {path}...")
    return picked_dataset(
        path,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )


if __name__ == "__main__":
    dataloader = get_dataloader(train=True, picked_dataset=dset.CIFAR10)
    image_tensors_test, _ = next(iter(dataloader))
    # display_images(
    #     image_tensors=image_tensors,
    #     batch_size=batch_size,
    #     image_size=image_size,
    #     device=get_device(),
    # )
    display_images_v2(
        image_tensors=image_tensors_test,
        device=get_device(),
    )
