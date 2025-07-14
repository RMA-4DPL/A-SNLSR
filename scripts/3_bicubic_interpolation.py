# Let's compute the statistics of bicubic interpolation

# We check the dataloader output
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from a_snlsr.data import SpectralDomain
from a_snlsr.data.datasets import TileDatasetAbundance
from a_snlsr.models.metrics import compute_metrics_torch
from a_snlsr.models.reports import generate_report_sisr
from a_snlsr.utils.device import load_device

device = load_device()

data_path = Path.cwd().parent / "data/"
dataset_paths = [data_path / "swir_park_dom1_x4", data_path / "swir_park_dom2_x4"]
splits_path = data_path / "swir1_sisr_swir2_val.npz"

domains = [SpectralDomain.SWIR_1, SpectralDomain.SWIR_2]
domain_indices = [0, 1]
report_path = Path.cwd().parent / "reports" / "bicubic_interpolation"


def compute_val_statistics(dataloader: DataLoader) -> tuple:
    min_value = torch.inf
    max_value = -torch.inf

    for batch in tqdm(
        dataloader, total=len(dataloader), desc="Computing stats on val dataset..."
    ):
        target = batch["hr_hsi"]

        min_value = min(min_value, target.min().item())
        max_value = max(max_value, target.max().item())

    return min_value, max_value


class BicubicModel(nn.Module):

    def __init__(
        self,
        super_resolution_factor: int = 4,
        max_val: float = 1.0,
        min_val: float = 0.0,
    ):
        super(BicubicModel, self).__init__()
        self.super_resolution_factor = super_resolution_factor
        self.max_val = max_val
        self.min_val = min_val

        self.layer = (
            nn.Identity()
        )  # No learnable parameters, just a bicubic upsampling layer

    def forward(self, x):
        return nn.functional.interpolate(
            x,
            scale_factor=self.super_resolution_factor,
            mode="bicubic",
            align_corners=False,
        )

    def inference(self, x, epoch=0):
        if isinstance(x, dict):
            x = x["lr_hsi"]

        return self.forward(x)

    def validate(self, dataloader, epoch=0):
        metrics = {}

        for batch in dataloader:
            input = batch["lr_hsi"]
            target = batch["hr_hsi"]

            output = self(input)

            loss = F.l1_loss(output, target)
            metrics["Loss"] = metrics.get("loss", 0) + loss.item()

            img_metrics = compute_metrics_torch(
                output,
                target,
                upsampling_ratio=self.super_resolution_factor,
                max_value=self.max_val,
                min_value=self.min_val,
            )
            for key, value in img_metrics.items():
                metrics[key] = metrics.get(key, 0) + value

        # Average the metrics
        for key in metrics:
            metrics[key] /= len(dataloader)

        return metrics["Loss"], metrics


if __name__ == "__main__":
    report_path.mkdir(parents=True, exist_ok=True)
    splits = np.load(splits_path, allow_pickle=True)["val"].item()

    dataset = TileDatasetAbundance(
        dataset_paths, domains, splits, validation=True, device=device
    )

    np.random.seed(2049)
    torch.manual_seed(2049)
    torch.cuda.manual_seed(2049)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    bicubic_stats = defaultdict(list)

    min_value, max_value = compute_val_statistics(dataloader)

    model = BicubicModel(4, max_value, min_value).to(device)

    generate_report_sisr(model, report_path, dataloader, spectral_domain=SpectralDomain.SWIR_2)  # type: ignore
