import datetime
import os
import torch
import rich

START_TIME = datetime.datetime.now()
BASE_DIR = os.getenv("SCRATCH", ".")
OUTPUT_DIR = os.path.join(f"{BASE_DIR}", "output")
SAVE_DIR = os.path.join(OUTPUT_DIR, START_TIME.strftime("%Y%m%d-%H%M%S"))
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")


def construct_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log(message: str):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    rich.print(f"{timestamp} {message}")
    with open(os.path.join(SAVE_DIR, "training.log"), "a", encoding="utf-8") as handle:
        rich.print(f"{timestamp} {message}", file=handle)


def convert_to_uint8(images):
    images = (images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    return images
