from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100


class Dataset(Dataset):
    def __init__(self, transform):
        self.dataset = CIFAR100("./data/cifar100", train=True, download=True)
        self.transform = transform
        self.num_samples = len(self.dataset)

    def __getitem__(self, idx):
        loaded_image, _image_class = self.dataset[idx]
        transformed_image = self.transform(loaded_image)
        return transformed_image

    def __len__(self):
        return self.num_samples
