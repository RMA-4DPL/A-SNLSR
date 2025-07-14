"""Utiliies to preprocess the hyperspectral datacubes."""

from pathlib import Path
from typing import Union

import numpy as np
import scipy.io as scio


def blur_downsample(
    img: np.ndarray, blur_kernel: np.ndarray, scale_factor: int
) -> np.ndarray:
    """
    Simulate the degradation to lr-hsi.

    From publication: W. -j. Guo, W. Xie, K. Jiang, Y. Li, J. Lei and L. Fang, "Toward Stable, Interpretable, and Lightweight Hyperspectral Super-Resolution," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023

    Official implementation: https://github.com/WenjinGuo/DAEM
    """
    height, width, bands = img.shape[0], img.shape[1], img.shape[2]
    kernel_size = blur_kernel.shape[0]
    if kernel_size != blur_kernel.shape[1]:
        raise Exception("Height and width of blur kernel should be equal")

    # Padding
    img_aligned = np.zeros(
        (height + kernel_size - scale_factor, width + kernel_size - scale_factor, bands)
    )
    img_aligned[
        (kernel_size - scale_factor) // 2 : height + (kernel_size - scale_factor) // 2,
        (kernel_size - scale_factor) // 2 : width + (kernel_size - scale_factor) // 2,
        :,
    ] = img

    # Only calculate the needed pixels
    img_result = np.zeros((height // scale_factor, width // scale_factor, bands))
    for i in range(height // scale_factor):
        for j in range(width // scale_factor):
            A = np.multiply(
                img_aligned[
                    i * scale_factor : i * scale_factor + kernel_size,
                    j * scale_factor : j * scale_factor + kernel_size,
                    :,
                ],
                blur_kernel[:, :, None],
            )
            A = np.sum(A, axis=0)
            A = np.sum(A, axis=0)
            img_result[i, j, :] = A

    return img_result


def spatial_downsampling(
    hypercube: np.ndarray, downsampling_factor: int = 4, progressive: bool = True
) -> Union[np.ndarray, tuple[np.ndarray, list[np.ndarray]]]:
    n_downsampling_steps = np.log2(downsampling_factor).astype(int)

    downsampled_cubes = []

    psf_kernel = scio.loadmat(Path(__file__).parent / "blur_kernel.mat")["data"]

    for down_idx in range(n_downsampling_steps):
        lr_hsi = blur_downsample(
            hypercube, psf_kernel, scale_factor=2 ** (down_idx + 1)
        )

        if down_idx < n_downsampling_steps - 1:
            downsampled_cubes.append(lr_hsi)

    if progressive:
        return lr_hsi, downsampled_cubes

    return lr_hsi
