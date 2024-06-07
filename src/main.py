import torch
import logging
import datetime
import torchvision.datasets as dset

from utils import (
    get_config,
    get_dataloader,
    get_model,
    save_generator_test,
    get_save_path,
)


def prepare_torch(seed: int):
    torch.manual_seed(int)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", config["device"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    modelIsLoaded = False

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = get_save_path(start_time, modelIsLoaded)

    config = get_config(save_path=save_path)
    prepare_torch(seed=config["seed"])

    train_dataloader = get_dataloader(
        config["batch_size"],
        image_size=config["image_size"],
        train=True,
        picked_dataset=dset.CIFAR10,
    )
    model = get_model(
        config=config,
        save_path=save_path,
        # save_name="ckpt_20240529-084155_epoch98.ckpt",
    )

    try:
        model.fit(
            train_dataloader,
            number_of_epochs=1000,
            generator_learning_rate=2e-4,
            discriminator_learning_rate=2e-4,
            save_images_frequency=1,
        )
    except KeyboardInterrupt:
        print("Ctrl + C detected")
    except Exception as e:
        print(f"Unforseen exception: {e}")
    finally:
        print(f"Saving model to {config['ckpt_save_path']}")
        model.save(f'{config["ckpt_save_path"]}model.pt', model.best_epoch)
        save_generator_test(config, model)
