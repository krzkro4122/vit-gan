from datetime import datetime
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as dset
from vitgan import ViTGAN

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.utils as vutils


CONFIG_PATH = f"{os.environ['HOME']}/rep/code/vit-gan/config.json"


def display_image(data_loader: DataLoader):
    examples = next(iter(data_loader))
    for labels, images in enumerate(examples):
        
        plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(img)
        plt.show()
        print(f"Label: {label}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Ten PC czyni CUDA!")
    else:
        print("Ten PC nie czyni CUDA'ow!")
    return device


def get_model(config, save_path: str, save_name: Optional[str] = None):
    model = ViTGAN(**config)
    if save_name:
        checkpoint_path = os.path.join(save_path, save_name)
        model.load(checkpoint_path)
        print(f"Loaded model from: {checkpoint_path}")
    return model


def get_config(save_path: str):
    with open(CONFIG_PATH, "rb") as f:
        config = json.load(f)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, CONFIG_PATH), "w", encoding="utf-8") as f:
        json.dump(config, f)

    config["ckpt_save_path"] = save_path
    writer = SummaryWriter(save_path)
    config["logger"] = writer
    return config


def save_generator_test(config, model):
    noise = torch.randn(
        config["img_size"], config["lattent_space_size"], device=config["device"]
    )
    fake = model.generate(noise)
    img = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(config["ckpt_save_path"], "fake.png"))


def get_dataloader(batch_size: int, image_size: int, train: bool, picked_dataset):
    dataset = get_dataset(
        image_size=image_size, train=train, picked_dataset=picked_dataset
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def get_dataset(image_size: int, train: bool, picked_dataset):
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
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )


def get_save_path(start_time: datetime, model_is_loaded: bool):
    return (
        os.path.join(f"{os.environ['SCRATCH']}/output", "20240529-084154_MNIST")
        if model_is_loaded
        else os.path.join(f"{os.environ['SCRATCH']}/output", start_time)
    )


if __name__ == "__main__":
    display_image(get_dataloader(4, 32, train=True, picked_dataset=dset.CIFAR10))
