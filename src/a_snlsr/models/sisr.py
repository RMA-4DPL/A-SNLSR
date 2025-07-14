"""Single-Image Super-Resolution Networks and Adversarial Super-Resolution Networks."""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from a_snlsr.logging import get_logger
from a_snlsr.models import SerializableNetwork
from a_snlsr.models.discriminator import Conv2dDomainDiscriminator
from a_snlsr.models.esrt import ESRT
from a_snlsr.models.metrics import compute_metrics_torch, compute_sam_torch
from a_snlsr.models.rcam import FusionSubnet
from a_snlsr.models.spdn import SSAM, SPDNEncoder, Upsample
from a_snlsr.models.srformer import SRFormer

logger = get_logger()


class SISRNetwork(SerializableNetwork, ABC):
    def __init__(
        self,
        hsi_bands: int,
        hsi_width: int,
        hsi_height: int,
        super_resolution_factor: int = 4,
        bicubic_skip: bool = False,
        val_dataset_statistics: Optional[dict] = None,
    ):
        super(SISRNetwork, self).__init__()

        self.hsi_bands = hsi_bands
        self.hsi_width = hsi_width
        self.hsi_height = hsi_height
        self.super_resolution_factor = super_resolution_factor
        self.bicubic_skip = bicubic_skip

        self.h_params.update(
            {
                "hsi_bands": hsi_bands,
                "hsi_width": hsi_width,
                "hsi_height": hsi_height,
                "super_resolution_factor": super_resolution_factor,
                "bicubic_skip": bicubic_skip,
            }
        )

        self.val_dataset_statistics = val_dataset_statistics

    def _to_device(self, inputs):
        if isinstance(inputs, dict):
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            inputs = [b.to(self._device) for b in inputs]
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self._device)
        else:
            raise ValueError(f"Cannot support type {type(inputs)} for input.")
        return inputs

    def _compute_val_statistics(self, dataloader: DataLoader) -> None:
        if self.val_dataset_statistics is not None:
            return

        min_value = torch.inf
        max_value = -torch.inf

        for batch in tqdm(
            dataloader, total=len(dataloader), desc="Computing stats on val dataset..."
        ):
            target = batch["hr_hsi"]

            min_value = min(min_value, target.min().item())
            max_value = max(max_value, target.max().item())

        self.val_dataset_statistics = {"min": min_value, "max": max_value}
        self.h_params.update({"val_dataset_statistics": self.val_dataset_statistics})

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, epoch: int = 0
    ) -> tuple[torch.Tensor, dict]:
        loss_dict = {}
        loss = F.l1_loss(outputs, targets, reduction="mean")
        loss_dict["Loss"] = loss.item()

        # Additional Validation metrics
        if not self.training:
            if self.val_dataset_statistics is None:
                raise RuntimeError("Validation dataset statistics weren't computed.")
            metrics = compute_metrics_torch(
                outputs,
                targets,
                upsampling_ratio=self.super_resolution_factor,
                max_value=self.val_dataset_statistics["max"],
                min_value=self.val_dataset_statistics["min"],
            )
            loss_dict.update(metrics)

        return loss, loss_dict

    def _forward_pass(self, batch: dict, epoch: int) -> dict:
        outputs = self(batch["lr_hsi"])
        targets = batch["hr_hsi"]

        loss, loss_dict = self.loss(outputs, targets, epoch=epoch)

        if self.training:
            loss.backward()

        return loss_dict

    def inference(self, batch: dict) -> torch.Tensor:
        self.eval()

        inputs = self._to_device(batch)["lr_hsi"]  # type: ignore

        with torch.no_grad():
            return self(inputs)

    def train_epoch(
        self, optimizer: Optimizer, dataloader: DataLoader, epoch: int = 0
    ) -> dict:
        self.train()

        epoch_stats = defaultdict(float)

        for batch in tqdm(
            dataloader, total=len(dataloader), desc=f"Training on epoch {epoch + 1}"
        ):
            batch = self._to_device(batch)

            optimizer.zero_grad()

            loss_dict = self._forward_pass(batch, epoch)
            for key, val in loss_dict.items():
                epoch_stats[f"Train/{key}"] += val

            optimizer.step()

        for key, val in epoch_stats.items():
            epoch_stats[key] = epoch_stats[key] / len(dataloader)

        return epoch_stats

    def validate(self, dataloader: DataLoader, epoch: int = 0) -> tuple[float, dict]:
        self.eval()

        self._compute_val_statistics(dataloader)

        epoch_stats = defaultdict(float)
        with torch.no_grad():
            for batch in tqdm(
                dataloader, total=len(dataloader), desc="Running validation..."
            ):
                batch = self._to_device(batch)

                loss_dict = self._forward_pass(batch, epoch)
                for key, val in loss_dict.items():
                    epoch_stats[f"Val/{key}"] += val

        for key, val in epoch_stats.items():
            epoch_stats[key] = epoch_stats[key] / len(dataloader)

        return epoch_stats["Val/Loss"], epoch_stats

    @abstractmethod
    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method must be implemented in subclasses.")


class RCAMSRNetwork(SISRNetwork):

    def __init__(
        self,
        hsi_bands: int,
        hsi_width: int,
        hsi_height: int,
        super_resolution_factor: int = 4,
        n_features: int = 64,
        n_dense: int = 4,
        n_rcam: int = 4,
        part_loss_alpha: float = 0.1,
        input_bands: Optional[int] = None,
        bicubic_skip: bool = False,
        val_dataset_statistics: Optional[dict] = None,
    ):
        super(RCAMSRNetwork, self).__init__(
            hsi_bands=hsi_bands,
            hsi_width=hsi_width,
            hsi_height=hsi_height,
            super_resolution_factor=super_resolution_factor,
            bicubic_skip=bicubic_skip,
            val_dataset_statistics=val_dataset_statistics,
        )

        self.upsample_steps = int(math.log2(super_resolution_factor))
        self.part_loss_alpha = part_loss_alpha
        self.bicubic_skip = bicubic_skip
        self.n_features = n_features
        self.n_dense = n_dense
        self.n_rcam = n_rcam
        self.input_bands = input_bands

        self.h_params.update(
            {
                "part_loss_alpha": part_loss_alpha,
                "n_features": n_features,
                "n_dense": n_dense,
                "n_rcam": n_rcam,
                "input_bands": input_bands,
            }
        )

        self.upsample_subnets = []

        for idx in range(self.upsample_steps):
            in_channels = (
                input_bands if (idx == 0 and input_bands is not None) else hsi_bands
            )
            # Create a FusionSubnet for each upsampling step
            self.upsample_subnets.append(
                FusionSubnet(
                    in_channels=in_channels,
                    out_channels=hsi_bands,
                    n_features=n_features,
                    upsample=2,
                    n_dense=n_dense,
                    n_rcam=n_rcam,
                )
            )

        self.upsample_subnets = nn.ModuleList(self.upsample_subnets)

        self.val_dataset_statistics = val_dataset_statistics

    def loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        partial_outputs: list,
        partial_targets: list,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        loss_dict = {}
        loss = F.l1_loss(outputs, targets, reduction="mean")
        # loss = F.mse_loss(outputs, targets, reduction="mean")
        loss_dict["Main Loss"] = loss.item()

        for idx, (part_output, part_target) in enumerate(
            zip(partial_outputs, partial_targets)
        ):
            partial_loss_stage = F.l1_loss(part_output, part_target, reduction="mean")
            # partial_loss_stage = F.mse_loss(part_output, part_target, reduction="mean")
            loss += partial_loss_stage
            loss_dict[f"Partial Loss #{idx}"] = partial_loss_stage.item()

        loss_dict["Loss"] = loss.item()

        # Additional Validation metrics
        if not self.training:
            if self.val_dataset_statistics is None:
                raise RuntimeError("Validation dataset statistics weren't computed.")
            metrics = compute_metrics_torch(
                outputs,
                targets,
                upsampling_ratio=self.upsample_steps**2,
                max_value=self.val_dataset_statistics["max"],
                min_value=self.val_dataset_statistics["min"],
            )
            loss_dict.update(metrics)

        return loss, loss_dict

    def _forward_pass(self, batch: dict, epoch: int) -> dict:
        input_hsi, target = batch["lr_hsi"], batch["hr_hsi"]
        partial_targets = [
            batch[key] for key in batch.keys() if key.startswith("lr_hsi_div")
        ]

        output, partial_outputs = self(input_hsi)

        loss, loss_dict = self.loss(
            output, target, partial_outputs, partial_targets, epoch=epoch
        )

        if self.training:
            loss.backward()

        return loss_dict

    def inference(self, batch: dict) -> torch.Tensor:
        """
        Perform inference on a batch of data.
        """
        self.eval()

        input_hsi = self._to_device(batch)["lr_hsi"]  # type: ignore

        # Perform forward pass without computing gradients
        with torch.no_grad():
            output, _ = self(input_hsi)

        return output

    def forward(self, hsi: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        original_hsi = hsi

        logger.layer_debug(
            "Received input HSI with shape: %s and MSI with shape: %s",
            hsi.shape,
        )

        subfusions = []

        for step in range(self.upsample_steps):
            logger.layer_debug(
                "Upsampling input HSI of shape %s",
                step,
                hsi.shape,
            )

            # Perform upsampling
            hsi = self.upsample_subnets[step](hsi)

            # We perform bicubic upsampling and add it to the output of the network, allowing the model to focus on the residuals
            if self.bicubic_skip:
                bicubic_hsi = F.interpolate(
                    original_hsi,  # type: ignore
                    size=(
                        self.hsi_height * 2 ** (step + 1),
                        self.hsi_width * 2 ** (step + 1),
                    ),
                    mode="bicubic",
                    align_corners=False,
                )
                hsi += bicubic_hsi

            if step < self.upsample_steps - 1:
                subfusions.append(hsi)

            logger.layer_debug(
                "After upsampling step %s, HSI has shape: %s", step, hsi.shape
            )

        # Run the last fusion network
        logger.layer_debug("Final fusion output shape: %s", hsi.shape)

        return hsi, subfusions


class SNLSR(SISRNetwork):
    def __init__(
        self,
        hsi_bands: int,
        hsi_width: int,
        hsi_height: int,
        n_features: int = 64,
        n_materials: int = 16,
        super_resolution_factor: int = 4,
        weight_spectral_loss: float = 0.1,
        val_dataset_statistics: Optional[dict] = None,
        bicubic_skip: bool = True,  # use of bicubic skip connection in the network architecture
        pretrain: bool = False,
    ):
        super(SNLSR, self).__init__(
            hsi_bands=hsi_bands,
            hsi_width=hsi_width,
            hsi_height=hsi_height,
            super_resolution_factor=super_resolution_factor,
            bicubic_skip=bicubic_skip,  # use of bicubic skip connection in the network architecture
            val_dataset_statistics=val_dataset_statistics,
        )

        self.h_params.update(
            {
                "n_features": n_features,
                "n_materials": n_materials,
                "weight_spectral_loss": weight_spectral_loss,
            }
        )

        self.weight_spectral_loss = weight_spectral_loss

        self.act = torch.nn.Tanh()
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.SPDN = SPDNEncoder(
            input_channels=hsi_bands,
            n_materials=n_materials,
        )
        self.endmember = nn.Conv2d(n_materials, hsi_bands, kernel_size=1, bias=False)
        # self.endmember = SPDNDecoder(output_channels=hsi_bands, n_materials=n_materials)

        self.SRhead = nn.Conv2d(n_materials, n_features, kernel_size=3, padding=1)

        self.block1 = SSAM(n_features, 1, 1)
        self.block2 = SSAM(n_features, 0, 2)
        self.block3 = SSAM(n_features, 0, 3)

        self.block4 = SSAM(n_features, 1, 1)
        self.block5 = SSAM(n_features, 0, 2)
        self.block6 = SSAM(n_features, 0, 3)

        self.Up = nn.Sequential(
            nn.Conv2d(n_features * 4, n_features, kernel_size=1),
            Upsample(scale=self.super_resolution_factor // 2, n_features=n_features),
        )
        self.Up2 = nn.Sequential(
            nn.Conv2d(n_features * 4, n_features, kernel_size=1),
            Upsample(scale=self.super_resolution_factor // 2, n_features=n_features),
            nn.Conv2d(n_features, n_materials, kernel_size=3, padding=1),
        )

        self.setup_pretrain(pretrain)

    def setup_pretrain(self, pretrain: bool = True) -> None:
        self.pretrain = pretrain
        # Freezes the layers if pretrain is set to False
        self.SPDN.requires_grad_(pretrain)
        self.endmember.requires_grad_(pretrain)

    def loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        rec_inputs: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        loss_dict = {}

        spatial_loss_upscaling = F.l1_loss(outputs, targets, reduction="mean")
        loss_dict["Spatial Loss Upscaling"] = spatial_loss_upscaling.item()

        spatial_loss_unmixing = F.l1_loss(rec_inputs, inputs, reduction="mean")
        loss_dict["Spatial Loss Unmixing"] = spatial_loss_unmixing.item()

        # Compute the reconstruction loss
        spectral_loss_upscaling = compute_sam_torch(outputs, targets)
        loss_dict["Spectral Loss Upscaling"] = spectral_loss_upscaling.item()

        spectral_loss_unmixing = compute_sam_torch(rec_inputs, inputs)
        loss_dict["Spectral Loss Unmixing"] = spectral_loss_unmixing.item()

        # Combine the losses
        loss = (spatial_loss_unmixing + spatial_loss_upscaling) + (
            self.weight_spectral_loss
            * (spectral_loss_unmixing + spectral_loss_upscaling)
        )

        loss_dict["Loss"] = loss.item()

        # Additional Validation metrics
        if not self.training:
            if self.val_dataset_statistics is None:
                raise RuntimeError("Validation dataset statistics weren't computed.")

            # Compute additional metrics for upscaling
            metrics = compute_metrics_torch(
                outputs,
                targets,
                upsampling_ratio=self.super_resolution_factor,
                max_value=self.val_dataset_statistics["max"],
                min_value=self.val_dataset_statistics["min"],
            )
            for key, value in metrics.items():
                loss_dict[f"Upscaling {key}"] = value

            # Compute additional metrics for unmixing
            metrics_unmixing = compute_metrics_torch(
                rec_inputs,
                inputs,
                upsampling_ratio=self.super_resolution_factor,
                max_value=self.val_dataset_statistics["max"],
                min_value=self.val_dataset_statistics["min"],
            )
            for key, value in metrics_unmixing.items():
                loss_dict[f"Unmixing {key}"] = value

        return loss, loss_dict

    def _forward_pass(self, batch: dict, epoch: int) -> dict:
        input_hsi, target = batch["lr_hsi"], batch["hr_hsi"]

        output, rec_input = self(input_hsi)

        loss, loss_dict = self.loss(output, target, input_hsi, rec_input, epoch=epoch)
        if self.training:
            loss.backward()

        return loss_dict

    def inference(self, batch: dict) -> torch.Tensor:
        """
        Perform inference on a batch of data.
        """
        self.eval()

        input_hsi = self._to_device(batch)["lr_hsi"]  # type: ignore

        # Perform forward pass without computing gradients
        with torch.no_grad():
            output, _ = self(input_hsi)

        return output

    def forward(self, hsi: torch.Tensor, return_abundance: bool = False) -> Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        abu = self.SPDN(hsi)
        abundances = abu

        rec_input = self.endmember(abu)

        abu = self.SRhead(abu)
        abu1 = self.block1(abu)
        abu2 = self.block2(torch.cat([abu, abu1], dim=1))
        abu3 = self.block3(torch.cat([abu, abu1, abu2], dim=1))

        tempAbu = self.Up(torch.cat([abu, abu1, abu2, abu3], dim=1))

        abu4 = self.block4(tempAbu)
        abu5 = self.block5(torch.cat([tempAbu, abu4], dim=1))
        abu6 = self.block6(torch.cat([tempAbu, abu4, abu5], dim=1))

        Abu = self.Up2(torch.cat([tempAbu, abu4, abu5, abu6], dim=1))

        SR = self.endmember(
            Abu
        )  # We do not want to backpropagate through the endmember layer during training

        if self.bicubic_skip:
            bicubic_hsi = F.interpolate(
                hsi,
                size=(
                    self.hsi_height * self.super_resolution_factor,
                    self.hsi_width * self.super_resolution_factor,
                ),
                mode="bicubic",
                align_corners=False,
            )
            SR += bicubic_hsi

        if not return_abundance:
            return SR, rec_input

        return SR, rec_input, abundances  # type: ignore


class ESRTNetwork(SISRNetwork):

    def __init__(
        self,
        hsi_bands: int,
        hsi_width: int,
        hsi_height: int,
        super_resolution_factor: int = 4,
        bicubic_skip: bool = False,
        val_dataset_statistics: Optional[dict] = None,
    ):
        super(ESRTNetwork, self).__init__(
            hsi_bands=hsi_bands,
            hsi_width=hsi_width,
            hsi_height=hsi_height,
            super_resolution_factor=super_resolution_factor,
            bicubic_skip=bicubic_skip,
            val_dataset_statistics=val_dataset_statistics,
        )

        self.esrt = ESRT(
            upscale=super_resolution_factor,
            channels=hsi_bands,
            bicubic_skip=bicubic_skip,
        )

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        return self.esrt(hsi)


class SRFormerNetwork(SISRNetwork):

    def __init__(
        self,
        hsi_bands: int,
        hsi_width: int,
        hsi_height: int,
        super_resolution_factor: int = 4,
        bicubic_skip: bool = False,
        val_dataset_statistics: Optional[dict] = None,
    ):
        super(SRFormerNetwork, self).__init__(
            hsi_bands=hsi_bands,
            hsi_width=hsi_width,
            hsi_height=hsi_height,
            super_resolution_factor=super_resolution_factor,
            bicubic_skip=bicubic_skip,
            val_dataset_statistics=val_dataset_statistics,
        )

        self.srformer = SRFormer(
            img_size=hsi_height,
            in_chans=hsi_bands,
            upscale=super_resolution_factor,
            bicubic_skip=bicubic_skip,
        )

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        return self.srformer(hsi)


class AdversarialSNLSR(SNLSR):

    def __init__(
        self,
        hsi_bands: int,
        hsi_width: int,
        hsi_height: int,
        n_features: int = 64,
        n_materials: int = 16,
        super_resolution_factor: int = 4,
        weight_spectral_loss: float = 0.1,
        discriminator_loss_ratio: float = 0.01,
        val_dataset_statistics: Optional[dict] = None,
        bicubic_skip: bool = True,
        n_domains: int = 2,  # Number of domains for the discriminator
        pretrain: bool = True,  # Whether to pretrain the model
    ):
        super(AdversarialSNLSR, self).__init__(
            hsi_bands=hsi_bands,
            hsi_width=hsi_width,
            hsi_height=hsi_height,
            n_features=n_features,
            n_materials=n_materials,
            super_resolution_factor=super_resolution_factor,
            weight_spectral_loss=weight_spectral_loss,
            val_dataset_statistics=val_dataset_statistics,
            bicubic_skip=bicubic_skip,
            pretrain=pretrain,
        )
        # self.discriminator = FCDomainDiscriminator(n_materials=n_materials, n_domains=n_domains, loss_ratio=0.01, n_features=16)  # Placeholder for discriminator
        self.discriminator = Conv2dDomainDiscriminator(
            n_materials=n_materials, n_domains=n_domains, loss_ratio=0.01, n_features=16
        )  # Placeholder for discriminator
        self.discriminator_loss_ratio = discriminator_loss_ratio

    def forward(self, X):
        # We compute the abundance map from the SNLSR model
        output, reconstruction, abundance = super(AdversarialSNLSR, self).forward(
            X, return_abundance=True
        )

        # We compute the discriminator output
        domain_prediction = self.discriminator(abundance)

        return output, reconstruction, abundance, domain_prediction

    def inference(self, batch: dict) -> torch.Tensor:
        """
        Perform inference on a batch of data.
        """
        self.eval()

        input_hsi = self._to_device(batch)["lr_hsi"]

        # Perform forward pass without computing gradients
        with torch.no_grad():
            output, _, _, _ = self(input_hsi)

        return output

    def loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        input_hsi: torch.Tensor,
        rec_input: torch.Tensor,
        domain_prediction: torch.Tensor,
        domain_target: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        loss_dict = {}

        if len(output) > 0:
            spatial_loss_upscaling = F.l1_loss(output, target, reduction="mean")
            spectral_loss_upscaling = compute_sam_torch(output, target)
        else:
            spatial_loss_upscaling = torch.tensor(0.0, device=output.device)
            spectral_loss_upscaling = torch.tensor(0.0, device=output.device)

        spatial_loss_unmixing = F.l1_loss(rec_input, input_hsi, reduction="mean")
        spectral_loss_unmixing = compute_sam_torch(rec_input, input_hsi)

        discriminator_loss = F.cross_entropy(
            domain_prediction, domain_target, reduction="mean"
        )

        snlsr_loss = (
            spatial_loss_upscaling
            + (self.weight_spectral_loss * spectral_loss_upscaling)
            + spatial_loss_unmixing
            + (self.weight_spectral_loss * spectral_loss_unmixing)
        )
        loss = snlsr_loss + discriminator_loss
        display_loss = snlsr_loss + (self.discriminator_loss_ratio * discriminator_loss)

        loss_dict["Loss"] = display_loss.item()
        loss_dict["Class. Loss"] = discriminator_loss.item()

        # Compute the accuracy on the discriminator predictions
        predicted_domains = torch.argmax(F.softmax(domain_prediction, dim=1), 1)
        correct_domains = (
            (predicted_domains == torch.argmax(domain_target, 1)).sum().item()
        )
        accuracy = correct_domains / (
            domain_target.size(0) * domain_target.size(2) * domain_target.size(3)
        )
        loss_dict["Class. Accuracy"] = accuracy

        if not self.training:
            performance_metrics = compute_metrics_torch(
                output,
                target,
                upsampling_ratio=self.super_resolution_factor,
                max_value=self.val_dataset_statistics["max"],
                min_value=self.val_dataset_statistics["min"],
            )
            loss_dict.update(performance_metrics)

        return loss, loss_dict

    def _forward_pass(self, batch: dict, epoch: int) -> dict:
        input_hsi, target = batch["lr_hsi"], batch["hr_hsi"]
        domain_targets = batch["domain_targets"]

        output, rec_input, _, domain_prediction = self(input_hsi)

        if self.training:
            # We remove the gradients of the high-resolution output for the SWIR2 domain
            swir1_batches = domain_targets[:, 0] == 1
            output = output[swir1_batches]
            target = target[swir1_batches]

        domain_targets = domain_targets[:, :, None, None].repeat(
            (1, 1, input_hsi.size(2), input_hsi.size(3))
        )
        loss, loss_dict = self.loss(
            output,
            target,
            input_hsi,
            rec_input,
            domain_prediction,
            domain_targets,
            epoch=epoch,
        )

        if self.training:
            loss.backward()

        return loss_dict
