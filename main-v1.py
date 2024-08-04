import torch
import logging
import torchvision.datasets as dset

from src.v1.config import config
from src.v1.utils import (
    get_dataloader,
    get_model,
    save_generator_test,
)


def prepare_torch(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        config.device = "cuda"
    print("Device: ", config.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prepare_torch(seed=config.seed)

    train_dataloader = get_dataloader(
        train=True,
        picked_dataset=dset.CIFAR10,
    )
    model = get_model(
        # save_name="ckpt_20240529-084155_epoch98.ckpt",
    )

    try:
        model.fit(
            train_dataloader,
            number_of_epochs=1000,
            save_images_frequency=1,
        )
    except KeyboardInterrupt:
        print("Ctrl + C detected")
    finally:
        print(f"Saving model to {config.ckpt_save_path}")
        model.save(f"{config.ckpt_save_path}model.pt", model.best_epoch)
        save_generator_test(model)
