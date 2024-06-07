from discriminator import Discriminator
from generator import Generator
from gan import PytorchGAN


class ViTGAN(PytorchGAN):
    def __init__(
        self,
        image_size,
        number_of_channels,
        lattent_space_size,
        generator_params=None,
        discriminator_params=None,
        criterion="bce",
        logger=None,
        optimizer="adam",
        device="cpu",
        ckpt_save_path=None,
        tag="",
        **kwargs
    ):
        """
        Main VitGAN class for this project
        :param image_size: images size, the image must be square sized
        :param number_of_channels: number of channel of the images
        :param lattent_space_size: umber of features in the lattent space
        :param generator_params: kwargs for optional parameters of the Generator, mandatory args will be filled automatically
        :param discriminator_params: kwargs for optional parameters of the Discriminator, mandatory args will be filled automatically
        :param criterion: loss used for training, BCE or MSE
        :param logger: tensorboard logger
        :param optimizer: optimizer to use for training
        :param device: cpu or cuda
        :param ckpt_save_path: save path for training checkpoints
        :param tag: model tag for saved file names
        """
        super().__init__(
            criterion=criterion,
            logger=logger,
            optimizer=optimizer,
            device=device,
            ckpt_save_path=ckpt_save_path,
            tag=tag,
        )

        self.image_size = image_size
        self.number_of_channels = number_of_channels
        self.lattent_space_size = lattent_space_size

        self.generator_params = {} if generator_params is None else generator_params
        self.discriminator_params = (
            {} if discriminator_params is None else discriminator_params
        )

        self.generator_params = generator_params
        self.discriminator_params = discriminator_params

        (
            self.generator_params["image_size"],
            self.generator_params["number_of_channels"],
            self.generator_params["lattent_size"],
        ) = (self.image_size, self.number_of_channels, self.lattent_space_size)
        (
            self.discriminator_params["image_size"],
            self.discriminator_params["number_of_channels"],
            self.discriminator_params["output_size"],
        ) = (self.image_size, self.number_of_channels, 1)

        # Necessary attributes for PytorchGAN
        self.generator = Generator(**self.generator_params)
        self.discriminator = Discriminator(**self.discriminator_params)
        self.generator_input_shape = (self.lattent_space_size,)

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.to(self.device)

    def forward(self, x):
        pass

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, imgs):
        return self.discriminator(imgs)
