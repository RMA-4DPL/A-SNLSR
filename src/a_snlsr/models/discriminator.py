from abc import ABC

import torch
import torch.nn as nn
from pytorch_revgrad import RevGrad


class FlattenChannels(nn.Module):
    """Flattens the channels of a 4D image-type tensor (batch_size, n_channels, height, width)
    to a 2D tensor (batch_size * height * width, n_channels).
    """

    def __init__(self):
        super(FlattenChannels, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_channels, _, _ = x.size()
        x = x.permute(0, 2, 3, 1)
        return x.reshape(-1, n_channels)


class ReshapeChannels(nn.Module):
    """Reshapes a 2D tensor (batch_size * width * height, n_channels)
    to a 4D image-type tensor (batch_size, n_channels, width, height)."""

    def __init__(self, width: int, height: int):
        super(ReshapeChannels, self).__init__()
        self.width = width
        self.height = height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples, n_channels = x.size()

        batch_size = n_samples // (self.width * self.height)
        x = x.view(batch_size, self.width, self.height, n_channels)
        return x.permute(
            0, 3, 1, 2
        )  # Change to (batch_size, n_channels, width, height)


class ReshapeDecoderLayer(nn.Module):
    """Reshapes for 1D Conv"""

    def __init__(self, input_channels: int, dim_size: int):
        super(ReshapeDecoderLayer, self).__init__()
        self.input_channels = input_channels
        self.dim_size = dim_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_channels, self.dim_size)
        return x


class DomainDiscriminator(ABC, nn.Module):

    def __init__(
        self,
        loss_ratio: float = 0.01,
        n_materials: int = 16,
        n_domains: int = 2,
        *args,
        **kwargs,
    ):
        super(DomainDiscriminator, self).__init__()
        self.h_params = {
            "loss_ratio": loss_ratio,
            "n_domains": n_domains,
            "n_materials": n_materials,
        }
        self.loss_ratio = torch.tensor(loss_ratio, requires_grad=False)
        self.n_domains = n_domains
        self.n_materials = n_materials

    def adapt_loss_ratio(self, loss_ratio: float):
        """Adapt the loss ratio of the discriminator."""
        self.loss_ratio = loss_ratio
        for layer in self.modules():
            if isinstance(layer, RevGrad):
                layer._alpha = torch.tensor(loss_ratio, requires_grad=False)
                break
        else:
            raise ValueError("No RevGrad layer found in the discriminator.")


class FCDomainDiscriminator(DomainDiscriminator):

    def __init__(
        self,
        n_materials: int,
        n_domains: int = 2,
        loss_ratio: float = 0.01,
        n_features: int = 8,
        dropout_ratio: float = 0.2,
        *args,
        **kwargs,
    ):
        super(FCDomainDiscriminator, self).__init__(loss_ratio, n_materials, n_domains)

        self.layers = nn.Sequential(
            RevGrad(loss_ratio),
            FlattenChannels(),
            nn.Linear(n_materials, n_features, bias=False),
            nn.BatchNorm1d(n_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(n_features, n_features * 2, bias=False),
            nn.BatchNorm1d(n_features * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(n_features * 2, self.n_domains),
            ReshapeChannels(32, 32),
        )

    def forward(self, X):
        # Reshape (batch_size, n_materials, height, width) to (batch_size, n_materials)
        return self.layers(X)


class ConvDomainDiscriminator(DomainDiscriminator):

    def __init__(
        self,
        n_materials: int,
        n_domains: int = 2,
        loss_ratio: float = 0.01,
        n_features: int = 8,
        dropout_ratio: float = 0.2,
        *args,
        **kwargs,
    ):
        super(ConvDomainDiscriminator, self).__init__(
            loss_ratio, n_materials, n_domains
        )

        self.layers = nn.Sequential(
            RevGrad(loss_ratio),
            nn.Conv1d(1, n_features, kernel_size=3, bias=False),
            nn.BatchNorm1d(n_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout_ratio),
            nn.Conv1d(n_features, n_features * 2, kernel_size=3, bias=False),
            nn.BatchNorm1d(n_features * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_ratio),
            nn.Flatten(),
            nn.Linear((n_materials - 4) * n_features * 2, n_domains),
        )

    def forward(self, X):
        batch_size, materials, height, width = X.shape
        X = X.permute(0, 2, 3, 1)
        X = X.view(
            batch_size * height * width, materials
        )  # Flatten the spatial dimensions
        X = X.unsqueeze(1)  # Add a channel dimension for Conv1d

        X = self.layers(X)

        # Reshape back to (batch_size, n_domains, height, width)
        X = X.view(batch_size, height, width, self.n_domains)
        X = X.permute(
            0, 3, 1, 2
        )  # Change back to (batch_size, n_domains, height, width)
        return X


class Conv2dDomainDiscriminator(DomainDiscriminator):
    def __init__(
        self,
        n_materials: int,
        n_domains: int = 2,
        loss_ratio: float = 0.01,
        n_features: int = 16,
        dropout_ratio: float = 0.2,
        *args,
        **kwargs,
    ):
        super(Conv2dDomainDiscriminator, self).__init__(
            loss_ratio, n_materials, n_domains
        )

        self.n_materials = n_materials
        self.n_domains = n_domains

        self.cnn = nn.Sequential(
            RevGrad(loss_ratio),
            nn.Conv2d(n_materials, n_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(n_features, n_features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # Downsample by 2 again
            nn.Dropout2d(dropout_ratio),
            nn.AdaptiveAvgPool2d(1),  # Output shape: (batch, n_features*2, 1, 1)
            nn.Flatten(),  # (batch, n_features*2)
            nn.Linear(n_features * 2, n_domains),
        )

    def forward(self, X):
        # X: (batch_size, n_materials, height, width)
        batch_size, _, height, width = X.shape
        out = self.cnn(X)  # (batch_size, n_domains)
        out = out.unsqueeze(-1).unsqueeze(-1)  # (batch_size, n_domains, 1, 1)
        out = out.expand(-1, -1, height, width)  # Repeat guess over spatial dims
        return out
