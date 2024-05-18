from constants import Constants
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100


class Dataset(Dataset):
    def __init__(self, transform):
        self.dataset = CIFAR100(Constants.DATA_DIRECTORY, train=True, download=True)
        self.transform = transform
        self.number_of_samples = len(self.dataset)

    def __getitem__(self, index):
        loaded_image, _image_class = self.dataset[index]
        transformed_image = self.transform(loaded_image)
        return transformed_image

    def __len__(self):
        return self.number_of_samples
