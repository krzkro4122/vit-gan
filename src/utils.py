from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T


def display_image(data_loader: DataLoader):
    examples = next(iter(data_loader))
    for label, img in enumerate(examples):
        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        print(f"Label: {label}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Ten PC czyni CUDA!")
    else:
        print("Ten PC nie czyni CUDA'ow!")
    return device


def l2normalize(v, eps=1e-4):
    return v / (v.norm() + eps)


def count_params(model):
    cpt = 0
    for x in model.parameters():
        cpt += x.numel()
    return cpt
