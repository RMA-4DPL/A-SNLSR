"""Model performance metrics computation functions"""

from typing import Union

import numpy as np
import torch
from torchmetrics.image import (
    ErrorRelativeGlobalDimensionlessSynthesis,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

TensorLike = Union[torch.Tensor, np.ndarray]


def compute_psnr_torch(
    out: torch.Tensor, target: torch.Tensor, max_value: float, min_value: float
) -> torch.Tensor:
    """Compute the PSNR between multiple images and targets using PyTorch."""
    psnr = PeakSignalNoiseRatio(data_range=max_value - min_value).to(out.device)
    return psnr(out, target).mean()


def compute_sam_torch(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot_product = torch.sum(out * target, dim=1)
    norm_out = torch.norm(out, dim=1) + 1e-4
    norm_target = torch.norm(target, dim=1) + 1e-4
    sam = torch.acos(dot_product / (norm_out * norm_target))
    return torch.mean(sam)


def compute_rmse_torch(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean square error, the square root of MSE using PyTorch."""
    return torch.sqrt(torch.mean((out - target) ** 2))


def compute_ssim_torch(
    out: torch.Tensor, target: torch.Tensor, max_value: float, min_value: float
) -> torch.Tensor:
    """Structural Similarity Index (SSIM) using PyTorch."""
    ssim = StructuralSimilarityIndexMeasure(data_range=max_value - min_value).to(
        out.device
    )
    return ssim(out, target).mean()


def compute_ergas_torch(
    out: torch.Tensor, target: torch.Tensor, upsampling_ratio: int
) -> torch.Tensor:
    """Relative Dimensionless Global Error in Synthesis (ERGAS) using PyTorch."""
    metric = ErrorRelativeGlobalDimensionlessSynthesis(ratio=upsampling_ratio).to(
        out.device
    )
    out = out.clamp(min=0, max=1)  # Ensure values are
    target = target.clamp(min=0, max=1)  # Ensure values are in [0, 1] range
    return metric(out, target)


def compute_metrics_torch(
    out: torch.Tensor,
    target: torch.Tensor,
    upsampling_ratio: int,
    max_value: float,
    min_value: float,
) -> dict:
    return {
        "PSNR": compute_psnr_torch(out, target, max_value, min_value).item(),
        "SAM": compute_sam_torch(out, target).item(),
        "RMSE": compute_rmse_torch(out, target).item(),
        "SSIM": compute_ssim_torch(out, target, max_value, min_value).item(),
        "ERGAS": compute_ergas_torch(
            out + 0.5, target + 0.5, upsampling_ratio
        ).item(),  # Adding 0.5 for distribution shift. Note: ERGAS should not be used as an absolute metric, as it is heavily dependant on the target distribution, but rather for relative comparisons.
    }
