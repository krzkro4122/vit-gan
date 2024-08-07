import datetime
import os
import torch
import rich
import numpy as np
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3


START_TIME = datetime.datetime.now()
BASE_DIR = os.getenv("SCRATCH", ".")
OUTPUT_DIR = f"{BASE_DIR}/output"
SAVE_DIR = os.path.join(OUTPUT_DIR, START_TIME.strftime("%Y%m%d-%H%M%S"))
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
NOISE_DIR = os.path.join(SAVE_DIR, "noise")


def construct_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
    if not os.path.exists(NOISE_DIR):
        os.mkdir(NOISE_DIR)


def log(message: str):
    rich.print(message)
    with open(os.path.join(SAVE_DIR, "training.log"), "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


class ToTensorUInt8(object):
    def __call__(self, pic):
        # Make the array writable by copying it
        img = np.array(pic, np.uint8, copy=True)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).contiguous()  # Convert to CxHxW
        return img


def convert_to_uint8(images):
    images = (images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    return images


def calculate_inception_score(images, batch_size=32, splits=10):
    # Convert images to float32 and normalize to [0, 1]
    images = images.float() / 255.0
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(images.device)
    with torch.no_grad():
        preds = model(images)[0]
    preds = F.softmax(preds, dim=1)
    scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits : (i + 1) * len(preds) // splits, :]
        kl_div = part * (torch.log(part) - torch.log(part.mean(dim=0, keepdim=True)))
        kl_div = kl_div.sum(dim=1)
        scores.append(kl_div.mean().exp().item())
    return torch.tensor(scores).mean(), torch.tensor(scores).std()
