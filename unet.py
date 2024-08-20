from typing import Tuple

import itertools

import torch
import torch.nn as nn


# the asbstract module for sub modules in the UNet model
class UNetModule(nn.Module):
    def __init__(self):
        super().__init__()

    # forward method with x, time embedding and promote embedding
    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        promote_embedding: torch.Tensor,
    ):
        raise NotImplementedError()


class MultiheadAttention2D(UNetModule):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, time_embedding, promote_embedding):
        # x.shape = batch, channel, height and width
        b, c, h, w = x.shape
        # reshape x to batch, height * width, channel
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        y, _ = self.mha(x, x, x)
        # reshape back to batch, channel, height, width
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return y


# Resnet Sub Module in UNet
class ResnetModule(UNetModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        kernel_size: int,
        padding: int = 1,
        num_groups: int = 8,
        is_residual: bool = True,
    ):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

        self.conv_2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

        if in_channels != out_channels:
            self.conv_3 = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        else:
            self.conv_3 = nn.Identity()

        # make promote and time embedding to be the same channels as the out_channels
        self.promote_fc = nn.Sequential(
            nn.Linear(embedding_dim, out_channels),
        )

        self.time_fc = nn.Sequential(
            nn.Linear(embedding_dim, out_channels),
        )

        self.is_residual = is_residual

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        promote_embedding: torch.Tensor,
    ):
        h = self.conv_1(x)
        time_embedding = self.time_fc(time_embedding)
        time_embedding = time_embedding.unsqueeze(-1).unsqueeze(-1)

        # if is_residual is True, we add promote_embedding to the output
        # and add a residual connection
        if self.is_residual:
            promote_embedding = self.promote_fc(promote_embedding)
            promote_embedding = promote_embedding.unsqueeze(-1).unsqueeze(-1)
            conv_res = self.conv_2(h + time_embedding + promote_embedding)
            return self.conv_3(x) + conv_res
        else:
            return self.conv_2(h + time_embedding)


# The down/middle/up layer of UNet
# Reference: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/U-net.png
class UNetLayer(UNetModule):
    def __init__(
        self,
        channels: Tuple[int],
        embedding_dim,
        kernel_size,
        num_of_mha,  # number of multiheadattention to add
        is_residual=True,
        first_in_channels_expand=1,
        num_heads=4,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.models = nn.ModuleList()
        for i in range(0, len(channels) - 1):
            expand = first_in_channels_expand if i == 0 else 1
            self.models.append(
                ResnetModule(
                    in_channels=channels[i] * expand,
                    out_channels=channels[i + 1],
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    is_residual=is_residual,
                )
            )
            if i < num_of_mha:
                self.models.append(
                    MultiheadAttention2D(
                        embed_dim=channels[i + 1], num_heads=self.num_heads
                    )
                )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        promote_embedding: torch.Tensor,
    ):
        for model in self.models:
            x = model(x, time_embedding, promote_embedding)
        return x


class UNet(UNetModule):
    num_groups = 8
    stride = 1
    # it seems the number of resnet per layer MUST be 2
    # otherwise the code may went wrong
    resnet_per_layer: int = 2

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int,
        hidden_channels: Tuple[int],  # (128, 256, 512) define the depth of the UNet
        # which layer we will apply the multiheadattention
        attention_layer_indices: Tuple[int] = (1,),
        kernel_size: int = 3,
    ):
        super().__init__()
        # by default
        # for a 2D convolution, the size formula is:
        # Math: size_{out} = \frac{size_{in} + 2\cdot padding - dilation\cdot(kernel\_size-1) - 1}{stride} + 1
        # so if we want
        # Math: size_{out} = size_{in}
        # we need
        # Math: padding = kernel\_size // 2 \cdot dilation, which is by default is 1
        # where kernelv_size should be an odd number, stride is 1
        padding = kernel_size // 2

        # in layer to make the image to have the same channels as the first hidden layer
        self.in_layer = nn.Conv2d(
            in_channels, hidden_channels[0], kernel_size, padding=padding
        )

        # UNet struct contains down blocks, middle blocks and up block
        # They follow different patterns, so we need to define them separately
        # And the down blocks has a bridge to connect the up blocks like briges

        # start to creat the down blocks
        # append the first lays to be the hidden_channels[0]
        down_hidden_channels = hidden_channels[0:1] + hidden_channels
        self.down_layers = nn.ModuleList()
        for i in range(len(down_hidden_channels) - 1):
            num_of_mha = self.resnet_per_layer if i in attention_layer_indices else 0
            self.down_layers.append(
                UNetLayer(
                    (
                        down_hidden_channels[i],
                        down_hidden_channels[i + 1],
                        down_hidden_channels[i + 1],
                    ),
                    embedding_dim,
                    kernel_size,
                    num_of_mha,
                )
            )
        self.avg_pool_2d_modules = nn.ModuleList()
        # we dont not add the last avg_pool_2d block
        for _ in range(len(down_hidden_channels) - 2):
            self.avg_pool_2d_modules.append(nn.AvgPool2d(kernel_size=2))
        # end to create the down blocks

        # start to create middle blocks
        num_of_mha = 1 if len(attention_layer_indices) > 0 else 0
        self.middle_layers = UNetLayer(
            (
                down_hidden_channels[-1],
                down_hidden_channels[-1],
                down_hidden_channels[-1],
            ),
            embedding_dim,
            kernel_size,
            num_of_mha,
            # DO not add residual for middle layers
            is_residual=False,
        )
        # end to create the middle blocks

        # start to creat the up blocks
        up_hidden_channels = list(reversed(down_hidden_channels))
        self.up_layers = nn.ModuleList()
        for i in range(len(up_hidden_channels) - 1):
            num_of_mha = (
                self.resnet_per_layer
                if len(up_hidden_channels) - 2 - i in attention_layer_indices
                else 0
            )
            self.up_layers.append(
                UNetLayer(
                    (
                        up_hidden_channels[i],
                        up_hidden_channels[i + 1],
                        up_hidden_channels[i + 1],
                    ),
                    embedding_dim,
                    kernel_size,
                    num_of_mha,
                    first_in_channels_expand=2,  # first_in_channels_expand set 2 because of the bridge concat
                )
            )
        self.conv_transpose2d_modules = nn.ModuleList()
        # we dont not add the last avg_pool_2d block
        for i in range(len(up_hidden_channels) - 2):
            self.conv_transpose2d_modules.append(
                nn.ConvTranspose2d(
                    in_channels=up_hidden_channels[i + 1],
                    out_channels=up_hidden_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                )
            )
        # end to create the up blocks

        # creat the out layers
        self.out_layer = nn.Sequential(
            nn.GroupNorm(self.num_groups, hidden_channels[0]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels[0],
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        promote_embedding: torch.Tensor,
    ):
        x = self.in_layer(x)

        bridges = []
        # pair down and pool with the same depth
        for down_layer, pool_module in itertools.zip_longest(
            self.down_layers, self.avg_pool_2d_modules, fillvalue=None
        ):
            x = down_layer(x, time_embedding, promote_embedding)
            bridges.append(x)
            if pool_module is not None:
                x = pool_module(x)

        x = self.middle_layers(x, time_embedding, promote_embedding)

        i = len(bridges) - 1
        # pair up and pool transpose2d with the same depth
        for up_layer, conv2d_module in itertools.zip_longest(
            self.up_layers, self.conv_transpose2d_modules, fillvalue=None
        ):
            # this is the bridges to connect the down and up blocks
            x = up_layer(
                torch.cat([x, bridges[i]], 1), time_embedding, promote_embedding
            )
            if conv2d_module is not None:
                x = conv2d_module(x)
            i -= 1

        x = self.out_layer(x)

        return x
