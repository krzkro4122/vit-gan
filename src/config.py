import os
import datetime
from typing import Any
from pydantic import BaseModel


def get_save_path(start_time: datetime, model_is_loaded: bool):
    return (
        os.path.join(f"{os.environ['SCRATCH']}/output", "20240529-084154_MNIST")
        if model_is_loaded
        else os.path.join(f"{os.environ['SCRATCH']}/output", start_time)
    )


modelIsLoaded = False
_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVE_PATH = get_save_path(_start_time, modelIsLoaded)


class EncoderParameters(BaseModel):
    patch_size: int = 8
    overlap: int = 2
    dropout_rate: float = 0.0


class MappingMLPParameters(BaseModel):
    layers: list[Any] = []
    activation: str = "gelu"
    dropout_rate: float = 0.0
    input_features: int | None = None
    output_features: int | None = None


class TransformerParameters(BaseModel):
    number_of_heads: int = 4
    attention_dropout_rate: float = 0.2
    mlp_layers: list[Any] = []
    mlp_activation: str = "relu"
    mlp_dropout: float = 0.2
    input_features: int | None = None
    spectral_scaling: bool | None = None
    lp: int | None = None


class GeneratorParameters(BaseModel):
    feature_hidden_size: int = 384
    number_of_transformer_layers: int = 4
    output_hidden_dimension: int = 768
    learning_rate: float = 2e-5


class DiscriminatorParameters(BaseModel):
    number_of_transformer_layers: int = 4
    encoder_params: EncoderParameters = EncoderParameters()
    transformer_params: TransformerParameters = TransformerParameters()
    mapping_mlp_params: MappingMLPParameters = MappingMLPParameters()
    learning_rate: float = 2e-5


class ViTGANParameters(BaseModel):
    seed: int = 0
    betas: tuple[float, float] = (0.5, 0.999)
    number_of_channels: int = 3
    image_size: int = 32
    batch_size: int = 128
    lattent_space_size: int = 1024
    device: str = "cpu"
    ckpt_save_path: str = SAVE_PATH
    generator_params: GeneratorParameters = GeneratorParameters()
    discriminator_params: DiscriminatorParameters = DiscriminatorParameters()


config = ViTGANParameters()
