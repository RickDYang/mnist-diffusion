from typing import Tuple

import math

import torch
import torch.nn as nn

from unet import UNet


# Sinusoidal Position Embedding for transformer
# Reference:
# https://github.com/Jackson-Kang/Pytorch-Diffusion-Model-Tutorial.git
# https://github.com/lucidrains/denoising-diffusion-pytorch
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int, device, max_length: int = 10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        inv_freq = torch.exp(
            torch.arange(half_dim, device=device)
            * (-math.log(max_length))
            / (half_dim - 1)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_embedding: torch.Tensor):
        pos_enc = torch.ger(time_embedding, self.inv_freq)
        pos_enc = torch.cat((pos_enc.sin(), pos_enc.cos()), dim=-1)
        return pos_enc


# pylint: disable=not-callable
# To predict this noise based on the corrupted image
# This is the major model/task of diffusion model
class Denoiser(nn.Module):
    position_embedding_max_length = 1000

    def __init__(
        self,
        n_classs: int,
        image_channels: int,
        embedding_layers: Tuple[int],
        hidden_channles: Tuple[int],  # (128, 256, 512)
        kernel_size: int,
        device,
    ):
        super().__init__()

        # hidden_channles[0] is used to make the time embedding and the first conv layer
        # to have the same channels, so that the two can be added together
        self.backbone = UNet(
            embedding_dim=embedding_layers[-1],
            in_channels=image_channels,
            hidden_channels=hidden_channles,
            kernel_size=kernel_size,
        )

        # embedding the diffusion time step and promotes
        # We use the same embedding layers for both time and promotes
        # if you want to different embedding layers, you can change the code here
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(
                embedding_layers[0],
                device=device,
                max_length=self.position_embedding_max_length,
            )
        )
        for i in range(1, len(embedding_layers)):
            self.time_embedding.append(
                nn.Linear(embedding_layers[i - 1], embedding_layers[i])
            )
            self.time_embedding.append(nn.ReLU())

        self.promotes_embedding = nn.Sequential(
            nn.Embedding(n_classs, embedding_layers[0])
        )
        for i in range(1, len(embedding_layers)):
            self.promotes_embedding.append(
                nn.Linear(embedding_layers[i - 1], embedding_layers[i])
            )
            self.promotes_embedding.append(nn.ReLU())

    # to predict this noise based on the corrupted image
    def forward(self, perturbed_x, time_t, promotes):
        # perturbed_x: is the input with noise
        # time_t: is the diffusion time step
        time_embedding = self.time_embedding(time_t)
        promotes_embedding = self.promotes_embedding(promotes)

        # forward the backbone UNet model
        y = self.backbone(perturbed_x, time_embedding, promotes_embedding)
        # y's shape is same as perturbed_x's after go through the UNet backbone
        return y


class VarianceSchedular:
    def __init__(
        self,
        beta_range: Tuple[float],  # (1e-4, 0.02))
        image_shape_length: Tuple[int],
        steps: int,
        device,
    ):
        self.device = device
        self.steps = steps
        self.image_shape_length = image_shape_length
        # linear variance schedule
        betas = torch.linspace(
            start=beta_range[0], end=beta_range[1], steps=steps, device=device
        )
        # generate reversed diffusion time-step
        self.reversed_steps = torch.arange(
            self.steps - 1, -1, -1, device=self.device
        ).reshape(self.steps, 1)

        # Calculate the variance schedule parameters cache to resue
        # Math: \sqrt{\beta}
        self.sqrt_betas = torch.sqrt(betas)
        # Math: \alpha=1-\beta
        self.alphas = 1 - betas
        # Math: \sqrt{1-\beta}
        self.sqrt_alphas = torch.sqrt(self.alphas)
        # Math: \overline\alpha_i = \prod_{j=0}^i(1-\beta_j)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # Math: \sqrt{\overline\alpha}
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        # Math: \sqrt{1-\overline\alpha}
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    # gather the corresponding value from the cache by time step
    def extract(self, values: torch.Tensor, t: torch.Tensor):
        out = values.gather(-1, t)
        # reshape to (batch_size, 1, 1, 1, 1, ...)
        output_size = (t.shape[0],) + ((1,) * self.image_shape_length)
        y = out.reshape(*output_size)
        return y

    def noise_at_t(self, x: torch.Tensor, t: torch.Tensor):
        # make noise
        # Math: \epsilon_t
        epsilon = torch.randn_like(x, device=self.device)

        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t)

        # Forward process with fixed variance schedule
        # Math: X_t = \sqrt{\overline\alpha_t}\cdot X_0 + \sqrt{1-\overline\alpha_t}\cdot \epsilon_t
        noisy_sample = x * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        return noisy_sample.detach(), epsilon

    def denoise(self, denoiser: nn.Module, x_t: torch.Tensor, promotes: torch.Tensor):
        # the reverse diffusion process
        for t in self.reversed_steps:
            x_t = self.denoise_at_t(denoiser, x_t, t, promotes)
        # now t = 0, we get the restored image
        return x_t

    def denoise_at_t(
        self, denoiser: nn.Module, x_t: torch.Tensor, t: torch.Tensor, promotes: torch.Tensor
    ):
        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_t = denoiser(x_t, t, promotes)

        alpha = self.extract(self.alphas, t)
        sqrt_alpha = self.extract(self.sqrt_alphas, t)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t)
        sqrt_beta = self.extract(self.sqrt_betas, t)

        z_t = torch.randn_like(x_t, device=self.device)
        # denoise at time t, utilizing predicted noise
        # pylint: disable=line-too-long
        # Math: X_{t-1} = \frac{1}{\sqrt{\alpha_t}}\cdot (X_t - \frac{1-\alpha_t}{\sqrt{1-\overline\alpha_t}}\cdot \epsilon_t) + \sqrt{\beta_t}\cdot z_t
        x_t_minus_1 = (
            1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_t)
            + sqrt_beta * z_t
        )

        return x_t_minus_1.clamp(-1.0, 1.0)


# denoising diffusion probabilistic model
class DDPM(nn.Module):
    def __init__(
        self,
        image_resolutions: Tuple[int],
        n_class: int,
        embedding_layers: Tuple[int],
        hidden_channels: Tuple[int],
        kernel_size: int,
        betas: Tuple[float],
        n_times: int,
        device=None,
    ):
        super().__init__()
        image_channels = image_resolutions[0]
        self.device = device
        self.n_times = n_times
        self.image_resolutions = image_resolutions
        self.n_class = n_class

        self.model = Denoiser(
            n_class,
            image_channels,
            embedding_layers,
            hidden_channels,
            kernel_size,
            device=device,
        )

        self.schedular = VarianceSchedular(
            betas, len(image_resolutions), n_times, device=device
        )

    def forward(
        self,
        x,  # the input image: (batch, channels, height, width)
        promotes,  # the digits tensor: (batch, )
    ):
        # scale x to (-1, 1)
        x = x * 2 - 1
        bacth_size = x.shape[0]
        # (1) randomly choose a diffusion time step and make some noise
        t = torch.randint(
            low=0, high=self.n_times, size=(bacth_size,), device=self.device
        ).long()
        # (2) perturb image with fixed variance schedule
        # return perturbed image and noise
        perturbed_images, epsilon = self.schedular.noise_at_t(x, t)
        # (3) forward diffusion process with perturbed image, random steps and promote(digist)
        # promote is to instruct the model to generate the image of the digit
        # return the predicted noise
        pred_epsilon = self.model(perturbed_images, t, promotes)
        return perturbed_images, epsilon, pred_epsilon

    def decode(self, promotes):
        x_t = torch.randn(1, *self.image_resolutions, device=self.device)
        # concatenate the input with the one-hot encoded digits
        x_0 = self.schedular.denoise(self.model, x_t, promotes)

        # scale x to (0, 1)
        img = (x_0 + 1) * 0.5
        return img
