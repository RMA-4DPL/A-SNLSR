"""Spectral Response Function"""

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np


@dataclass
class LinearSRF:
    # Begin of the spectrum, in nanometers
    begin_nm: float = 1000.0

    # End of the spectrum, in nanometers
    end_nm: float = 1000.0

    # Spectral steps
    spectral_steps: int = 300

    # Output bands after integration
    n_bands: int = 3

    # Matrix representing the STF, of size spectral_steps x n_bands
    values: Optional[np.ndarray] = None

    # Wavelength coordinates
    spectral_centers: np.ndarray = None

    output_spectral_centers: np.ndarray = None

    def apply(self, datacube: np.ndarray) -> np.ndarray:
        return np.tensordot(datacube, self.values, axes=([2], [0]))

    def band_cutoff(
        self, hypercube: np.ndarray, cutoff_band_ranges: list
    ) -> np.ndarray:
        hypercube = hypercube.copy()
        changed_srf = self.copy()

        # Find the indices in the cutoff ranges
        spectral_centers = changed_srf.spectral_centers

        select_array = np.ones_like(spectral_centers, dtype=bool)

        for start_cutoff, end_cutoff in cutoff_band_ranges:
            start_index = np.argmin(np.abs(spectral_centers - start_cutoff))
            end_index = np.argmin(np.abs(spectral_centers - end_cutoff))
            select_array[start_index:end_index] = False

        hypercube = hypercube[:, :, select_array]

        # Updated the SRF
        changed_srf.values = changed_srf.values[select_array, :]
        changed_srf.spectral_centers = changed_srf.spectral_centers[select_array]
        changed_srf.spectral_steps = changed_srf.values.shape[0]
        changed_srf.begin_nm = changed_srf.spectral_centers[0]
        changed_srf.end_nm = changed_srf.spectral_centers[-1]

        return hypercube, changed_srf

    def copy(self) -> "LinearSRF":
        return copy.deepcopy(self)

    def serialize_mat(self) -> dict:
        """Returns a `serialized` version of this object compatible with they key/array style for MATLAB records."""
        return {
            "begin_nm": self.begin_nm,
            "end_nm": self.end_nm,
            "spectral_steps": self.spectral_steps,
            "n_bands": self.n_bands,
            "values": self.values,
            "spectral_centers": self.spectral_centers,
        }

    @classmethod
    def deserialize_mat(mapping: dict) -> "LinearSRF":
        return LinearSRF(**mapping)


def make_gaussian(x, mean, std):
    gaussian_vals = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )
    return gaussian_vals / np.sum(gaussian_vals)


def _validate_generate_srf_params(begin_nm: float, end_nm: float, spectral_steps: int):
    if begin_nm >= end_nm:
        raise ValueError("begin_nm should have a lower value than end_nm.")
    if spectral_steps < 0:
        raise ValueError(
            "Please provide a positive value for the number of spectral steps."
        )


def generate_srf_gaussian(
    means: np.ndarray,
    stds: np.ndarray,
    begin_nm: float = 1000.0,
    end_nm: float = 2500.0,
    spectral_steps: int = 300,
) -> LinearSRF:
    _validate_generate_srf_params(begin_nm, end_nm, spectral_steps)
    if len(means) != len(stds):
        raise ValueError(
            "Please provide the same amount of means and standard deviations."
        )

    band_srfs = []
    x_values = np.linspace(begin_nm, end_nm, spectral_steps)
    for mean, std in zip(means, stds):
        band_srfs.append(make_gaussian(x_values, mean, std))
    band_srfs = np.array(band_srfs, dtype=np.float32).T

    return LinearSRF(
        begin_nm=begin_nm,
        end_nm=end_nm,
        spectral_steps=spectral_steps,
        n_bands=len(means),
        values=band_srfs,
        spectral_centers=np.linspace(begin_nm, end_nm, spectral_steps),
        output_spectral_centers=means,
    )


def generate_srf_singleband(
    bands: np.ndarray,
    begin_nm: float = 1000.0,
    end_nm: float = 2500.0,
    spectral_steps: int = 300,
) -> LinearSRF:
    _validate_generate_srf_params(begin_nm, end_nm, spectral_steps)

    band_srfs = np.zeros((spectral_steps, len(bands)))

    for idx, band in enumerate(bands):
        band_srfs[band, idx] = 1.0

    return LinearSRF(
        begin_nm,
        end_nm,
        spectral_steps,
        len(bands),
        band_srfs,
        np.linspace(begin_nm, end_nm, spectral_steps),
        output_spectral_centers=bands,
    )


@lru_cache
def generate_srf_band_reduction(
    input_bands: int, output_bands: int, begin_nm: int = 1000, end_nm: int = 2500
):
    fwhm = (end_nm - begin_nm) / output_bands
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Estimation of std from FWHM

    return generate_srf_gaussian(
        means=np.linspace(begin_nm, end_nm, output_bands),
        stds=np.ones(output_bands) * std,
        spectral_steps=input_bands,
        begin_nm=begin_nm,
        end_nm=end_nm,
    )
