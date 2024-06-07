import os
import json
import torch
import logging
import datetime
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from vitgan import ViTGAN

CONFIG_PATH = f"{os.environ['HOME']}/rep/code/vit-gan/config.json"
DATASET_NAME = "CIFAR10"


def get_config():
    with open(CONFIG_PATH, "rb") as f:
        config = json.load(f)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, CONFIG_PATH), "w") as f:
        json.dump(config, f)

    config["ckpt_save_path"] = save_path
    writer = SummaryWriter(save_path)
    config["logger"] = writer

    return config


def save_generator_test(config):
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


def prepare_torch():
    torch.manual_seed(config["seed"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", config["device"])


def get_dataset():
    path = os.path.abspath(f"{os.environ['SCRATCH']}/data")
    return dset.CIFAR10(
        root=path,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(config["img_size"]),
                transforms.CenterCrop(config["img_size"]),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datasetIsLoaded = False

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = (
        os.path.join(f"{os.environ['SCRATCH']}/output", "20240529-084154_MNIST")
        if datasetIsLoaded
        else os.path.join(f"{os.environ['SCRATCH']}/output", start_time + f"_{DATASET_NAME}")
    )

    config = get_config()
    dataset = get_dataset()
    print(dataset)
    prepare_torch()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1
    )

    model = ViTGAN(**config)
    if datasetIsLoaded:
        checkpoint_path = os.path.join(save_path, "ckpt_20240529-084155_epoch98.ckpt")
        model.load(checkpoint_path)
        print(f"Loaded model from: {checkpoint_path}")
    try:
        model.fit(
            dataloader, n_epochs=1000, gen_lr=2e-4, disc_lr=2e-4, save_images_freq=1
        )
    except KeyboardInterrupt:
        print(f"Ctrl + C detected")
    except Exception as e:
        print(f"Unforseen exception: {e}")
    finally:
        print(f"Saving model to {config['ckpt_save_path']}")
        model.save(f'{config["ckpt_save_path"]}model.pt', model.best_epoch)
        save_generator_test(config)
