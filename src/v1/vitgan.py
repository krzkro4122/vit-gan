from src.v1.config import config
from src.v1.discriminatorViT import Discriminator
from src.v1.generator import Generator
from src.v1.gan import GAN


class ViTGAN(GAN):
    def __init__(self):
        super().__init__(generator=Generator(), discriminator=Discriminator())
        self.generator_input_shape = (config.lattent_space_size,)

        self.generator.to(config.device)
        self.discriminator.to(config.device)
        self.to(config.device)

    def forward(self, x):
        pass

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, imgs):
        return self.discriminator(imgs)
