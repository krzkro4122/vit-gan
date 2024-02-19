import torchvision

from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, filenames, transform):
        self.filenames = filenames
        self.transform = transform
        self.num_samples = len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = torchvision.io.read_image(filename)
        image = self.transform(image)
        return image

    def __len__(self):
        return self.num_samples
