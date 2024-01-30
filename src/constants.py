from dataclasses import dataclass


@dataclass
class Constants:
    NUMBER_OF_STEPS = 10000
    BATCH_SIZE = 128
    BETA1 = 0.0
    BETA2 = 0.99
    DATA_DIR = "data/bedroom_train/"
    SAMPLE_INTERVAL = 1000
    LATENT_VECTOR_DIMENSIONS = 1024
