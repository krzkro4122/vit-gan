from constants import Constants
from utils import get_dataloader, get_dataset
from torchvision.utils import make_grid
import torchvision.transforms as T
import torchvision


def data_test():
    dataset = get_dataset(dataset_path=Constants.DATA_DIRECTORY)
    images = [dataset[i] for i in range(128, 256)]
    grid = torchvision.utils.make_grid(images, nrow=4)
    torchvision.utils.save_image(grid, f"output/data_test.jpg")


def load_test():
    data_loader = get_dataloader(Constants.DATA_DIRECTORY)
    real_image = next(data_loader)
    sample = make_grid(real_image, nrow=4, normalize=True)
    sample = T.ToPILImage()(sample)
    sample.save(f"output/load_test.jpg")


if __name__ == "__main__":
    data_test()
    load_test()
