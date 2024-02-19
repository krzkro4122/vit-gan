from dataclasses import dataclass


@dataclass
class Constants:
    NUMBER_OF_STEPS = 100000
    SAMPLE_INTERVAL = NUMBER_OF_STEPS // 15
    BATCH_SIZE = 128
    LATENT_VECTOR_DIMENSIONS = 1024
    BETA1 = 0.0
    BETA2 = 0.99
    LEARNING_RATE = 0.002
    DATA_DIRECTORY = "data/bedroom_train/"
