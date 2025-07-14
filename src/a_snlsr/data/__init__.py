"""Data processing methods and data structures for HSI."""

import argparse
from enum import Enum

BAND_CUTOFF_RANGES_SWIR = [(1310, 1550), (1800, 1980)]
BAND_CUTOFF_RANGES_VNIR = [(0, 450)]
BAND_CUTOFF_RANGES_SWIR_1 = [(1310, 1550)]
BAND_CUTOFF_RANGES_SWIR_2 = [(1800, 1980)]

MSI_BANDS_SWIR_1 = [1174.3311036789298, 1348.6622073578596, 1522.9933110367892]
MSI_WIDTH_SWIR_1 = 120
MSI_BANDS_SWIR_2 = [1901.7558528428094, 2101.1705685618726, 2300.5852842809363]
MSI_WIDTH_SWIR_2 = 120

MSI_BANDS_SWIR = [1200, 1700, 2200]
MSI_WIDTH_SWIR = 300

MSI_BANDS_VNIR = [477, 716, 955]
MSI_WIDTH_VNIR = 130


class SpectralDomain(Enum):

    VNIR = (
        0,
        "VNIR",
        400,
        1000,
        MSI_BANDS_VNIR,
        MSI_WIDTH_VNIR,
        BAND_CUTOFF_RANGES_VNIR,
        1.0,
    )
    SWIR = (
        1,
        "SWIR",
        1000,
        2500,
        MSI_BANDS_SWIR,
        MSI_WIDTH_SWIR,
        BAND_CUTOFF_RANGES_SWIR,
        0.5,
    )
    SWIR_1 = (
        2,
        "SWIR_1",
        1000,
        1700,
        MSI_BANDS_SWIR_1,
        MSI_WIDTH_SWIR_1,
        BAND_CUTOFF_RANGES_SWIR_1,
        1.0,
    )
    SWIR_2 = (
        3,
        "SWIR_2",
        1700,
        2500,
        MSI_BANDS_SWIR_2,
        MSI_WIDTH_SWIR_2,
        BAND_CUTOFF_RANGES_SWIR_2,
        0.5,
    )

    def __init__(
        self,
        id: int,
        domain_name: str,
        begin_mm: float,
        end_mm: float,
        msi_bands: list,
        msi_band_width: int,
        cutoff_ranges: list,
        msi_alpha_correction: float,
    ):
        self.id = id
        self.domain_name = domain_name
        self.begin_nm = begin_mm
        self.end_nm = end_mm
        self.msi_bands = msi_bands
        self.msi_band_width = msi_band_width
        self.cutoff_ranges = cutoff_ranges
        self.msi_alpha_correction = msi_alpha_correction

    def __str__(self):
        return self.domain_name

    @classmethod
    def from_domain_name(cls, domain_name: str):
        for item in cls:
            if item.domain_name == domain_name:
                return item
        raise ValueError(f"Coudln't parse SpectralDomain with name: {domain_name}.")

    @classmethod
    def from_id(cls, domain_id: int):
        for item in cls:
            if item.id == domain_id:
                return item
        raise ValueError(f"Coudln't parse SpectralDomain with id: {domain_id}.")


class SpectralDomainAction(argparse.Action):
    """Helper class to parse a spectraldomain enum from the string value in CLI parameters."""

    def __init__(self, *args, **kwargs):
        super(SpectralDomainAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            enum_item = [SpectralDomain.from_domain_name(val) for val in values]
        else:
            enum_item = SpectralDomain.from_domain_name(values)  # type: ignore
        setattr(namespace, self.dest, enum_item)


__all__ = ["SpectralDomain", "SpectralDomainAction"]
