from datetime import datetime
from dataset import LSUNBedroomDataset, InfiniteSampler

import os
import glob
from torch.utils.data import DataLoader
import torchvision
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
    dataset = LSUNBedroomDataset(filenames, transform)
    return dataset


def get_dataloader(dataset_path, img_size=64, batch_size=64):
    data = get_dataset(dataset_path, img_size)
    data_loader = iter(
        DataLoader(
            data, batch_size=batch_size, num_workers=1, sampler=InfiniteSampler(data)
        )
    )
    return data_loader


def _test(dataset_path):
    dataset = get_dataset(dataset_path=dataset_path)
    print(dataset[1].shape)
    images = [dataset[i] for i in range(26, 42)]
    grid_image = torchvision.utils.make_grid(images, nrow=4)
    torchvision.utils.save_image(grid_image, "single_sample.jpg")
    print("Saved an image to single_sample.jpg")


def get_date_code():
    date = datetime.now()
    return f"{date.day}-{date.month}-{date.year}_{date.hour}-{date.minute}-{date.second}"
