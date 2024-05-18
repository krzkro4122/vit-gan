from dataclasses import dataclass
from datetime import datetime


def get_date_code():
    date = datetime.now()
    return (
        f"{date.day}-{date.month}-{date.year}_{date.hour}-{date.minute}-{date.second}"
    )


@dataclass(frozen=True)
class Constants:
    NUMBER_OF_STEPS = 10000
    SAMPLE_INTERVAL = NUMBER_OF_STEPS // 100
    BATCH_SIZE = 64
    LATENT_VECTOR_DIMENSIONS = 128
    BETA1 = 0.5
    BETA2 = 0.999
    LEARNING_RATE = 0.0002
    DATA_DIRECTORY = "data/cifar100/"
    OUTPUT_FOLDER_NAME = f"output/train_{get_date_code()}"
