"""Re-implementation and adaptation of the paper:

Tao Zhang, Ying Fu, Liwei Huang, Siyuan Li, Shaodi You, and Chenggang Yan, “Rgb-guided hyperspectral image superresolution with deep progressive learning,” CAAI Transactions on Intelligence Technology, vol. 9, no. 3, pp. 679–694, July 2023.

Originally conceived for fusion-based hyperspectral image super-resolution, but adapted for single image super-resolution.
"""

import torch
import torch.nn as nn

from a_snlsr.logging import get_logger

logger = get_logger()


class RCAM(nn.Module):
    """Residual channel attention module"""

    def __init__(
        self, in_channels: int, n_features: int = 64, attention_reduction_ratio: int = 4
    ):
        super().__init__()
        # Pre-computation of certain features.
        self.pre_attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=1),
        )
        # Attention mask computation.
        self.attention_computation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=n_features,
                out_channels=n_features // attention_reduction_ratio,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_features // attention_reduction_ratio,
                out_channels=n_features,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # We compute a map ready to be applied by attention
        X_pre = self.pre_attention(X)

        # We compute the attention
        attention_map = self.attention_computation(X_pre)

        # We apply the attention by multiplication
        post_attention = X_pre.mul(attention_map)

        # Skip connection for faster convergence.
        return post_attention + X_pre


class DenseBlock(nn.Module):
    """Contains multiple RCAM modules"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_features: int = 64,
        n_rcam: int = 4,
        post_rcam_ksize: int = 3,
        post_rcam_padding: int = 1,
        leaky_relu: bool = False,
    ):
        super().__init__()
        # Number of channels in the input as well as the output.s
        self.in_channels = n_features
        # Number of channels after all the rcam layers are concatenated
        self.rcam_layers_output = n_features * n_rcam
        # The last activation of the network should be a leaky relu network, or it can be stuck to learn some bands.
        self.leaky_relu = leaky_relu

        self.rcam_layers = nn.ModuleList(
            [
                RCAM(
                    in_channels=in_channels if layer_idx == 0 else n_features,
                    n_features=n_features,
                )
                for layer_idx in range(n_rcam)
            ]
        )

        # Post RCAM feature aggregation
        self.post_rcam = nn.Sequential(
            nn.Conv2d(
                in_channels=self.rcam_layers_output,
                out_channels=out_channels,
                kernel_size=post_rcam_ksize,
                padding=post_rcam_padding,
            ),
            nn.LeakyReLU() if leaky_relu else nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        rcam_outputs = []

        out_rcam = X
        for rcam_idx, rcam_layer in enumerate(self.rcam_layers):
            out_rcam = rcam_layer(out_rcam)
            rcam_outputs.append(out_rcam)
            logger.layer_debug(f"RCAM Layer N{rcam_idx} output shape: {out_rcam.shape}")

        # We concatenate the results of all the RCAM layers and compute a convolution reaching the output shape
        rcam_outputs = torch.concatenate(rcam_outputs, axis=1)  # type: ignore

        logger.layer_debug(f"Concatenated RCAM output layers: {rcam_outputs.shape}")

        output = self.post_rcam(rcam_outputs)

        logger.layer_debug(f"Post-rcam output shape: {output.shape}")
        return output


class FusionSubnet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_features: int,
        upsample: int = 2,
        n_dense: int = 4,
        n_rcam: int = 4,
    ):
        super().__init__()

        self.upsample = upsample

        self.pre_db = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=1),
            nn.ReLU(),
        )

        db_networks = []

        for idx in range(n_dense):
            post_rcam_ksize = 3 if idx != (n_dense - 1) else 1
            post_rcam_padding = 1 if idx != (n_dense - 1) else 0
            leaky_relu = False if idx != (n_dense - 1) else True
            db_out_channels = n_features if idx != (n_dense - 1) else out_channels

            db_networks.append(
                DenseBlock(
                    in_channels=n_features * (idx + 1),
                    out_channels=db_out_channels,
                    n_features=n_features,
                    n_rcam=n_rcam,
                    post_rcam_ksize=post_rcam_ksize,
                    post_rcam_padding=post_rcam_padding,
                    leaky_relu=leaky_relu,
                )
            )

        self.db_networks = nn.Sequential(*db_networks)

        if self.upsample > 1:
            self.upsample_layer = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.upsample * 2,
                stride=self.upsample,
                padding=self.upsample // 2,
                output_padding=0,
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pre_db = self.pre_db(X)

        db_output = None
        db_outputs = [pre_db]

        for db_idx, db in enumerate(self.db_networks):
            db_input = torch.concatenate(db_outputs, dim=1)
            logger.layer_debug(
                f"Input shape for DB network N{db_idx}: {db_input.shape}"
            )
            db_output = db(db_input)
            logger.layer_debug(
                f"Ouput shape for DB network N{db_idx}: {db_output.shape}"
            )
            db_outputs.append(db_output)
        if db_output is None:
            raise ValueError(
                "Network has no DB subnets, please set n_dense > 0 in networks's hyper paramters."
            )

        if self.upsample > 1:
            db_output = self.upsample_layer(db_output)

        return db_output
