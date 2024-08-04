import torch
from torch import nn

from src.v1.config import EncoderParameters, config


class PatchEncoder(nn.Module):
    def __init__(
        self,
        encoder_parameters: EncoderParameters,
    ):
        super().__init__()

        self.patch_size = encoder_parameters.patch_size
        self.overlap = encoder_parameters.overlap

        self.token_size = (
            config.number_of_channels * (self.patch_size + 2 * self.overlap) ** 2
        )
        self.stride = (
            config.image_size - self.patch_size - 2 * self.overlap
        ) // self.patch_size + 1
        self.number_of_tokens = (
            (config.image_size - (self.patch_size + 2 * self.overlap - 1) - 1)
            // self.stride
            + 1
        ) ** 2

        self.projection_matrix = nn.Linear(
            self.token_size, self.projection_output_size, bias=False
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.projection_output_size))
        self.positional_embedding = nn.Parameter(
            torch.randn(self.number_of_tokens + 1, self.projection_output_size)
        )

        self.dropout = nn.Dropout(p=encoder_parameters.dropout_rate)

    def forward(self, images):
        assert (
            len(images.shape) == 4
        ), "Expected input image tensor to be of shape BxCxHxW"
        images_tokens = self._get_tokens(images)
        images_tokens = self.projection_matrix(images_tokens)
        cls_token = self.cls_token.expand(
            images_tokens.shape[0], 1, self.projection_output_size
        )

        tokens = torch.cat((cls_token, images_tokens), dim=1)
        embedding_tokens = tokens + self.positional_embedding
        embedding_tokens = self.dropout(embedding_tokens)
        return embedding_tokens

    def _get_tokens(self, images):
        # To get the stride with overlap, we simulate a bigger patch, but accomodate only for a smaller one
        assert (
            images.shape[2] == images.shape[3]
        ), "The provided images are not square shaped"

        # Then we use the unfold method twice to get the batches with respect to both image dimension
        images_patches = images.unfold(
            2, self.patch_size + 2 * self.overlap, self.stride
        ).unfold(3, self.patch_size + 2 * self.overlap, self.stride)

        # We reshape this 5d tensor according in a BxLxF shape: we flatten patch and channel, and flatten their 2d positionning too
        # view as Batch size, n_batch_x * n_batch_y, channels * patch_h * patch_w
        images_patches = images_patches.contiguous()
        sequence_patches = images_patches.view(
            images_patches.shape[0],
            images_patches.shape[2] * images_patches.shape[3],
            images_patches.shape[1] * images_patches.shape[4] * images_patches.shape[5],
        )
        return sequence_patches
