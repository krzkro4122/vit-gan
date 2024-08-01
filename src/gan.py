import datetime
import os

import torch
from torch import nn
from typing import Any
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from src.config import config, SAVE_PATH


def pick_criterion(criterion: str):
    match criterion:
        case "mse":
            return nn.MSELoss(reduction="mean")
        case "bce":
            return nn.BCELoss(reduction="mean")
        case _:
            return criterion


class GAN(nn.Module):
    def __init__(
        self,
        criterion="bce",
        optimizer="adam",
        tag="",
    ):
        super().__init__()
        self.log = SummaryWriter(SAVE_PATH)

        self.optimizer_type = optimizer if optimizer in ["sgd", "adam"] else ValueError
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.state = {}
        self.criterion = pick_criterion(criterion=criterion)
        self.best_criterion = {
            "[TRAINING] Discriminator LOSS on real data": 10**10,
            "[TRAINING] Discriminator LOSS on fake data": 10**10,
            "[TRAINING] Discriminator LOSS total": 10**10,
            "[TRAINING] Generator LOSS": 10**10,
            "[VALIDATION] Discriminator LOSS on validation data": 10**10,
            "[VALIDATION] Discriminator LOSS on fake data": 10**10,
            "[VALIDATION] Discriminator LOSS total": 10**10,
            "[VALIDATION] Generator LOSS": 10**10,
            "Discriminator FID": 10**10,
        }
        self.best_model = None
        self.best_epoch = None

        # /!\ the overriding class must implement a discriminator and a generator extending nn.Module
        self.generator_input_shape = None
        self.generator: Any | None = None
        self.discriminator: Any | None = None

        # useful stuff that can be needed for during fit
        self.start_time = datetime.datetime.now()
        self.verbose = None
        self.number_of_epochs = None
        self.n = None
        self.tag = tag

        self.save_images_freq = None

    def _train_epoch(self, dataloader):
        fid = FID().to(config.device)
        fid_batch = torch.randint(0, len(dataloader), (20, 1))

        epoch_disc_real_loss = 0
        epoch_disc_fake_loss = 0
        epoch_disc_tot_loss = 0
        epoch_gen_loss = 0
        discriminator_fid = 0

        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(config.device)
            batch_size = batch_x.size(0)

            # Compute the loss the for the discriminator with real images
            self.discriminator.zero_grad()
            label = torch.full(
                (batch_size,), 1, dtype=torch.float, device=config.device
            )
            real_disc_out = self.discriminator(batch_x).view(-1)
            disc_real_loss = self.criterion(real_disc_out, label)
            disc_real_loss.backward()

            # Compute the loss the for the discriminator with fake images
            noise_shape = [batch_size] + list(self.generator_input_shape)
            noise = torch.randn(noise_shape, device=config.device)
            fake_images = self.generator(noise)
            label.fill_(0)  # changing the label
            fake_disc_out = self.discriminator(fake_images.detach()).view(
                -1
            )  # we do not backprop on the generator
            disc_fake_loss = self.criterion(fake_disc_out, label)
            disc_fake_loss.backward()

            disc_tot_loss = disc_real_loss + disc_fake_loss
            self.discriminator_optimizer.step()

            # Training the generator
            self.generator.zero_grad()
            label.fill_(1)  # for the generator, all image are real as we construct them
            out = self.discriminator(fake_images).view(
                -1
            )  # this time we want to backprop on the generator
            gen_loss = self.criterion(out, label)
            gen_loss.backward()
            self.generator_optimizer.step()

            if idx in fid_batch:
                with torch.no_grad():
                    fake_images_8b = (
                        ((fake_images + 1) * 255 / 2).to(torch.uint8).to(config.device)
                    )
                    batch_x_8b = (
                        ((batch_x + 1) * 255 / 2).to(torch.uint8).to(config.device)
                    )
                    fid.update(
                        fake_images_8b.expand(
                            (
                                fake_images_8b.shape[0],
                                3,
                                fake_images_8b.shape[2],
                                fake_images_8b.shape[3],
                            )
                        ),
                        real=False,
                    )
                    fid.update(
                        batch_x_8b.expand(
                            (
                                batch_x_8b.shape[0],
                                3,
                                batch_x_8b.shape[2],
                                batch_x_8b.shape[3],
                            )
                        ),
                        real=True,
                    )

            # Update running losses
            epoch_disc_real_loss += disc_real_loss.item()
            epoch_disc_fake_loss += disc_fake_loss.item()
            epoch_disc_tot_loss += disc_tot_loss.item()
            epoch_gen_loss += gen_loss.item()

        fid_value = fid.compute()

        return (
            epoch_disc_real_loss / len(dataloader),
            epoch_disc_fake_loss / len(dataloader),
            epoch_disc_tot_loss / len(dataloader),
            epoch_gen_loss / len(dataloader),
            fid_value / len(dataloader),
        )

    def _validate(self, dataloader):
        epoch_disc_real_loss = 0
        epoch_disc_fake_loss = 0
        epoch_disc_tot_loss = 0
        epoch_gen_loss = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(config.device)
            batch_size = batch_x.size(0)

            # Compute the loss the for the discriminator with real images
            self.discriminator.zero_grad()
            label = torch.full(batch_size, 1, dtype=torch.float, device=config.device)
            real_disc_out = self.discriminator(batch_x).view(-1)
            disc_real_loss = self.criterion(real_disc_out, label)

            # Compute the loss the for the discriminator with fake images
            noise_shape = [batch_size] + list(self.generator_input_shape)
            noise = torch.randn(noise_shape, device=config.device)
            fake_images = self.generator(noise)
            label.fill_(-1)  # changing the label
            fake_disc_out = self.discriminator(fake_images.detach()).view(
                -1
            )  # we do not backprop on the generator
            disc_fake_loss = self.criterion(fake_disc_out, label)

            disc_tot_loss = disc_real_loss + disc_fake_loss

            # Training the generator is not different now (no need to backprop)
            gen_loss = disc_fake_loss

            # Update running losses
            epoch_disc_real_loss += disc_real_loss.item()
            epoch_disc_fake_loss += disc_fake_loss.item()
            epoch_disc_tot_loss += disc_tot_loss.item()
            epoch_gen_loss += gen_loss.item()

        return (
            epoch_disc_real_loss / len(dataloader),
            epoch_disc_fake_loss / len(dataloader),
            epoch_disc_tot_loss / len(dataloader),
            epoch_gen_loss / len(dataloader),
        )

    def fit(
        self,
        dataloader,
        number_of_epochs,
        generator_learning_rate,
        discriminator_learning_rate,
        validation_data=None,
        verbose=1,
        save_images_frequency=50,
        save_criterion="Discriminator FID",
        ckpt=None,
        save_model_freq=50,
        betas=(0.5, 0.999),
        **kwargs,
    ):
        assert (
            self.generator is not None
        ), "Model does not seem to have a generator, assign the generator to the self.generator attribute"
        assert (
            self.discriminator is not None
        ), "Model does not seem to have a discriminator, assign the discriminator to the self.discriminator attribute"
        assert (
            self.generator_input_shape is not None
        ), "Could not find the generator input shape, please specify this attribute before fitting the model"

        if self.optimizer_type == "sgd":
            self.generator_optimizer = torch.optim.SGD(
                params=self.generator.parameters(), lr=generator_learning_rate
            )
            self.discriminator_optimizer = torch.optim.SGD(
                params=self.discriminator.parameters(), lr=discriminator_learning_rate
            )
        elif self.optimizer_type == "adam":
            self.generator_optimizer = torch.optim.Adam(
                params=self.generator.parameters(),
                lr=generator_learning_rate,
                betas=betas,
            )
            self.discriminator_optimizer = torch.optim.Adam(
                params=self.discriminator.parameters(),
                lr=discriminator_learning_rate,
                betas=betas,
            )
        else:
            raise ValueError("Unknown optimizer")

        start_epoch = 0
        self.verbose = verbose

        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state["epoch"]
            self.load_state_dict(state["state_dict"])
            for g in self.discriminator_optimizer.param_groups:
                g["lr"] = state["lr"]["disc_lr"]
            for g in self.generator_optimizer.param_groups:
                g["lr"] = state["lr"]["gen_lr"]

        self.number_of_epochs = number_of_epochs
        for n in range(start_epoch, number_of_epochs):
            self.n = n
            self.train()
            (
                t_disc_real_loss,
                t_disc_fake_loss,
                t_disc_total_loss,
                t_gen_loss,
                discriminator_fid,
            ) = self._train_epoch(dataloader)
            v_disc_real_loss, v_disc_fake_loss, v_disc_total_loss, v_gen_loss = (
                0,
                0,
                0,
                0,
            )
            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    (
                        v_disc_real_loss,
                        v_disc_fake_loss,
                        v_disc_total_loss,
                        v_gen_loss,
                    ) = self._validate(validation_data)

            epoch_result = {
                "[TRAINING] Discriminator LOSS on real data": t_disc_real_loss,
                "[TRAINING] Discriminator LOSS on fake data": t_disc_real_loss,
                "[TRAINING] Discriminator LOSS total": t_disc_total_loss,
                "[TRAINING] Generator LOSS": t_gen_loss,
                "[VALIDATION] Discriminator LOSS on validation data": v_disc_real_loss,
                "[VALIDATION] Discriminator LOSS on fake data": v_disc_real_loss,
                "[VALIDATION] Discriminator LOSS total": v_disc_total_loss,
                "[VALIDATION] Generator LOSS": v_gen_loss,
                "Discriminator FID": discriminator_fid,
            }
            if self.logger:
                for k, v in epoch_result.items():
                    self.logger.add_scalar(k, v, n)

            if epoch_result[save_criterion] <= self.best_criterion[save_criterion]:
                self.best_criterion = epoch_result
                self.__save_state(n)

            if n % verbose == 0:
                print(
                    "Epoch {:3d} Gen loss: {:1.4f} Disc loss: {:1.4f} Disc real loss {:1.4f} Disc fake loss {:1.4f} | Validation Gen loss: {:1.4f} Disc loss: {:1.4f} Disc real loss {:1.4f} Disc fake loss {:1.4f} | FID value {:1.4f} | Best epoch {:3d}".format(
                        n,
                        t_gen_loss,
                        t_disc_total_loss,
                        t_disc_real_loss,
                        t_disc_fake_loss,
                        v_gen_loss,
                        v_disc_total_loss,
                        v_disc_real_loss,
                        v_disc_fake_loss,
                        discriminator_fid,
                        self.best_epoch,
                    )
                )

            if save_images_frequency is not None and n % save_images_frequency == 0:
                noise = torch.randn(
                    config.image_size, config.lattent_space_size, device=config.device
                )
                image_tensors = self.generate(noise)
                tensors = [
                    make_grid(
                        image_tensor.to(config.device)[: config.image_size],
                        normalize=True,
                    ).cpu()
                    for image_tensor in image_tensors
                ]
                image_tensors_denormalized = torch.stack(tensors)
                image_tensors_grid = make_grid(image_tensors_denormalized)
                self.logger.add_image("images", image_tensors_grid, n)

            if save_model_freq is not None and n % save_model_freq == 0:
                assert config.ckpt_save_path is not None, "Need a path to save models"
                self.save(
                    {
                        "gen_lr": generator_learning_rate,
                        "disc_lr": discriminator_learning_rate,
                    },
                    n,
                )

        print(
            f'Training completed in {str(datetime.datetime.now() - start_time).split(".")[0]}'
        )

    def save(self, lr, n):
        self.state["lr"] = lr
        self.state["epoch"] = n
        self.state["state_dict"] = self.state_dict()
        if not os.path.exists(config.ckpt_save_path):
            os.mkdir(config.ckpt_save_path)
        torch.save(
            self.state,
            os.path.join(
                config.ckpt_save_path,
                f"ckpt_{self.start_time.strftime('%Y%m%d-%H%M%S')}_epoch{n}.ckpt",
            ),
        )

    def load(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.load_state_dict(state["state_dict"])

    def __save_state(self, n):
        self.best_epoch = n
        self.best_model = self.state_dict()

    def __load_saved_state(self):
        if self.best_model is None:
            raise ValueError("No saved model available")
        self.load_state_dict(self.best_model)
