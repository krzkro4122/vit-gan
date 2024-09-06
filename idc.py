import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import logging
import datetime

START_TIME = datetime.datetime.now()
BASE_DIR = os.getenv("SCRATCH", ".")
OUTPUT_DIR = os.path.join(f"{BASE_DIR}", "output")
SAVE_DIR = os.path.join(OUTPUT_DIR, START_TIME.strftime("%Y%m%d-%H%M%S"))
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
MODEL_DIR = os.path.join(SAVE_DIR, "model")

# Global Paths and Hyperparameters
DATA_PATH = OUTPUT_DIR
MODEL_PATH = MODEL_DIR
PLOT_PATH = OUTPUT_DIR
EPOCHS = 100
BATCH_SIZE = 64
LATENT_DIM = 128
FID_BEST = np.inf

# Setup pretty logging for both file and stdout
def setup_logging(log_dir):
    log_file = os.path.join(SAVE_DIR, f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

# CIFAR-10 Dataset loader
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root=DATA_PATH, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Vision Transformer Generator (ViT-GAN) Network
class ViTGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(ViTGenerator, self).__init__()
        self.latent_dim = latent_dim
        # Simplified Generator block
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.generator(z)
        return out.view(z.size(0), 3, 32, 32)

# Vision Transformer Discriminator
class ViTDiscriminator(nn.Module):
    def __init__(self):
        super(ViTDiscriminator, self).__init__()
        # Simplified Transformer block for discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.discriminator(x)

# Training Loop with Metric Tracking
def train(generator, discriminator, data_loader, optimizer_G, optimizer_D, device):
    global FID_BEST
    writer = SummaryWriter(log_dir=OUTPUT_DIR)
    for epoch in range(EPOCHS):
        for i, (imgs, _) in enumerate(data_loader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Training Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_imgs = generator(z)
            g_loss = nn.BCELoss()(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Training Discriminator
            optimizer_D.zero_grad()
            real_loss = nn.BCELoss()(discriminator(real_imgs), valid)
            fake_loss = nn.BCELoss()(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                logging.info(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
                writer.add_scalar("Loss/Generator", g_loss.item(), epoch * len(data_loader) + i)
                writer.add_scalar("Loss/Discriminator", d_loss.item(), epoch * len(data_loader) + i)

        # Generate and save example images after each epoch
        with torch.no_grad():
            gen_imgs = generator(z).detach().cpu()
        save_image(gen_imgs, epoch)

        # Check FID and save model if it's the best so far
        current_fid = calculate_fid()  # Placeholder for actual FID calculation
        if current_fid < FID_BEST:
            FID_BEST = current_fid
            save_model(generator, discriminator, epoch)

        logging.info(f"Epoch {epoch} - FID: {current_fid}")

    writer.close()

# Generation Tester
def test_generator(generator, n_images, device):
    generator.eval()
    z = torch.randn(n_images, LATENT_DIM, device=device)
    with torch.no_grad():
        gen_imgs = generator(z).detach().cpu()
    save_image(gen_imgs, "test")

# Save images as grid
def save_image(images, epoch):
    img_path = Path(PLOT_PATH) / f"epoch_{epoch}.png"
    vutils.save_image(images, img_path, normalize=True)

# Save model function
def save_model(generator, discriminator, epoch):
    torch.save(generator.state_dict(), os.path.join(MODEL_PATH, f"generator_epoch_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(MODEL_PATH, f"discriminator_epoch_{epoch}.pth"))

# Plot metrics such as FID, IS, loss
def plot_metrics(metrics):
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.savefig(os.path.join(PLOT_PATH, 'metrics.png'))

# Placeholder FID calculation
def calculate_fid():
    # Actual FID implementation would go here
    return np.random.rand() * 100  # Random placeholder FID

def main():
    setup_logging(OUTPUT_DIR)

    # Load data
    dataloader = get_dataloader(BATCH_SIZE)

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = ViTGenerator(LATENT_DIM).to(device)
    discriminator = ViTDiscriminator().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train models
    train(generator, discriminator, dataloader, optimizer_G, optimizer_D, device)

    # Test generator
    test_generator(generator, 10, device)

if __name__ == "__main__":
    main()
