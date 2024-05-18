from constants import Constants
from dataset import Dataset
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler
import glob
import os
import torch
import torchvision.transforms as T


def get_dataset(dataset_path, image_size=64):
    filenames = glob.glob(os.path.join(dataset_path, "*"))
    compose = [
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = T.Compose(compose)
    dataset = Dataset(filenames, transform)
    return dataset


def get_dataset_cifar100(image_size=64):
    compose = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = T.Compose(compose)
    return Dataset(transform=transform)


def get_dataloader(image_size=64):
    dataset_path = Constants.DATA_DIRECTORY
    batch_size = Constants.BATCH_SIZE
    dataset = get_dataset_cifar100(image_size=image_size)
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,
            sampler=RandomSampler(dataset, replacement=True, num_samples=int(1e10)),
        )
    )
    return data_loader


def create_output_directories():
    os.makedirs(Constants.OUTPUT_FOLDER_NAME, exist_ok=True)
    os.makedirs(f"{Constants.OUTPUT_FOLDER_NAME}/weights", exist_ok=True)
    os.makedirs(f"{Constants.OUTPUT_FOLDER_NAME}/samples", exist_ok=True)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "Ten PC nie czyni CUDA'ow!"
    return device
