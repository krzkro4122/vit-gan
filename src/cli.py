import argparse

from constants import Constants


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=Constants.NUMBER_OF_STEPS,
        help="Number of steps for training (Default: 100000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Constants.BATCH_SIZE,
        help="Size of each batches (Default: 128)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=Constants.LEARNING_RATE,
        help="Learning rate (Default: 0.002)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=Constants.BETA1,
        help="Coefficients used for computing running averages of gradient and its square",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=Constants.BETA2,
        help="Coefficients used for computing running averages of gradient and its square",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=Constants.LATENT_VECTOR_DIMENSIONS,
        help="Dimensions of the latent vector",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=Constants.DATA_DIR,
        help="Data root dir of your training data",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1000,
        help="Interval for sampling image from generator",
    )
    return parser