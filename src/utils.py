from datetime import datetime

from dataset import Dataset

import os
import glob
from torch.utils.data import DataLoader, RandomSampler
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


def get_dataloader(dataset_path, img_size=64, batch_size=64):
    dataset = get_dataset(dataset_path, img_size)
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,
            sampler=RandomSampler(dataset, replacement=True, num_samples=int(1e10)),
        )
    )
    return data_loader


def get_date_code():
    date = datetime.now()
    return (
        f"{date.day}-{date.month}-{date.year}_{date.hour}-{date.minute}-{date.second}"
    )
