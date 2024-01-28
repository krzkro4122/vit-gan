import torch
import torchvision

from torch.utils.data import Dataset, Sampler


class LSUNBedroomDataset(Dataset):
    def __init__(self, filenames, transform):
        self.filenames = filenames
        self.transform = transform
        self.num_samples = len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = torchvision.io.read_image(filename)
        # Resize and normalize the image
        image = self.transform(image)
        return image

    def __len__(self):
        return self.num_samples


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.N = len(data_source)

    def __iter__(self):
        while True:
            for idx in torch.randperm(self.N):
                yield idx
