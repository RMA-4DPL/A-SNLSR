"""Base class for an HSI image, refactoring in construction."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from a_snlsr.data import SpectralDomain
from a_snlsr.data.preprocessing import spatial_downsampling
from a_snlsr.data.srf import (
    LinearSRF,
    generate_srf_band_reduction,
    generate_srf_gaussian,
)
from a_snlsr.utils.misc import is_power_of_two


def lazy(method):
    """Decorator to automatically support lazy computation."""

    def wrapper(self, *args, **kwargs):
        new_data = self.data
        new_instance = self.copy(data=new_data)

        return method(new_instance, *args, **kwargs)

    return wrapper


class HSIDataArray(xr.DataArray):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs["band_cutoffs"] = []

    @lazy
    def cutoff_bands(self, start_band, end_band) -> "HSIDataArray":

        band_coords = self.coords["band"].values
        start_index = np.searchsorted(band_coords, start_band)
        end_index = np.searchsorted(band_coords, end_band, side="right")

        if start_index < 0 or end_index > len(band_coords):
            raise ValueError("Invalid band wavelengths for cutoff.")

        # Remove the bands from the coordiantes and values
        new_data = np.concatenate(
            (self.values[:, :, :start_index], self.values[:, :, end_index:]), axis=2
        )
        new_band_coords = np.concatenate(
            (band_coords[:start_index], band_coords[end_index:])
        )

        new_attrs = self.attrs.copy()
        new_attrs["band_cutoffs"].append((start_band, end_band))

        return HSIDataArray(
            data=new_data,
            dims=("x", "y", "band"),
            coords=dict(x=self.x, y=self.y, band=new_band_coords),
            attrs=new_attrs,
        )

    @lazy
    def spatial_downsample(
        self, downsampling_factor: int, progressive: bool = False
    ) -> "HSIDataArray":
        # Check if downsampling factor is a power of 2
        if not is_power_of_two(downsampling_factor):
            raise ValueError(
                f"Downsampling factor should be a power of two: {downsampling_factor} was provided."
            )

        if not progressive:
            new_data = spatial_downsampling(
                self.data, downsampling_factor, progressive=progressive
            )
        else:
            new_data, partial_arrays = spatial_downsampling(
                self.data, downsampling_factor, progressive=progressive
            )
            partial_hsi = [
                HSIDataArray.from_numpy(arr, self.start_band, self.end_band)
                for arr in partial_arrays
            ]

        self = HSIDataArray(
            data=new_data,
            dims=self.dims,
            coords=dict(
                x=np.arange(new_data.shape[0]),
                y=np.arange(new_data.shape[1]),
                band=self.band,
            ),
            attrs=self.attrs,
        )

        if not progressive:
            return self

        return self, partial_hsi

    @lazy
    def band_minmax_norm(
        self, min_val: Optional[np.ndarray] = None, max_val: Optional[np.ndarray] = None
    ) -> "HSIDataArray":
        if min_val is None:
            min_val = self.min(dim=("x", "y"))
        if max_val is None:
            max_val = self.max(dim=("x", "y"))

        if (len(min_val) != len(self.coords["band"])) or (
            len(max_val) != len(self.coords["band"])
        ):
            raise ValueError(
                "Provided min_val and max_val should have the same length as the number of bands.s"
            )

        return (self - min_val) / (max_val - min_val)

    @lazy
    def apply_srf(self, srf: LinearSRF) -> "HSIDataArray":
        data = srf.apply(self.data)
        self = HSIDataArray(
            data=data,
            dims=["x", "y", "band"],
            coords=dict(
                x=self.x,
                y=self.y,
                band=srf.output_spectral_centers,
            ),
            attrs=self.attrs,
        )
        return self

    @lazy
    def spectral_downsample(self, output_bands: int) -> "HSIDataArray":
        srf = generate_srf_band_reduction(
            input_bands=self.band.size,
            output_bands=output_bands,
            begin_nm=self.start_band,
            end_nm=self.end_band,
        )
        return self.apply_srf(srf)

    @lazy
    def compute_msi(self, means: list, std: float) -> "HSIDataArray":
        msi_srf = generate_srf_gaussian(
            means,
            np.ones(len(means)) * std,
            begin_nm=self.start_band,
            end_nm=self.end_band,
            spectral_steps=self.band.size,
        )
        return self.apply_srf(msi_srf)

    def plot(self, x, y, axis=None, figsize=None, **kwargs) -> plt.Figure:
        if axis is None:
            fig, axis = plt.subplots(figsize=figsize)

        pixel_spectrum = self.values[x, y]
        axis.plot(pixel_spectrum, **kwargs)
        axis.set_title(f"Spectrum for pixel ({x}, {y})")
        axis.set_xlabel("Wavelength [nm]")
        axis.set_ylabel("Intensity [AU]")

        # Add vertical dashed lines for cutoffs
        for start_band, _ in self.attrs["band_cutoffs"]:
            axis.axvline(x=start_band, color="r", linestyle="--")

        locator = MaxNLocator(integer=True)
        axis.xaxis.set_major_locator(locator)

        ticks = axis.get_xticks()
        tick_labels = [f"{int(tick)}" for tick in ticks if tick in self.band.values]
        axis.set_xticklabels(tick_labels)

        return fig

    def plot_interactive(
        self,
        means: Optional[list] = None,
        std: Optional[float] = None,
        figsize: tuple = None,
        gamma: float = 1.0,
        **kwargs,
    ) -> plt.Figure:
        fig, (ax_im, ax_plot) = plt.subplots(ncols=2, figsize=figsize)

        if self.band.size == 2 or (
            ((means is None) or (std is None)) and self.band.size > 3
        ):
            raise ValueError(
                "HSI Image should either have 1 (panchromatic), 3 (RGB) or N > 3 bands to be plotted. Please provide `means` and `std` if > 3."
            )

        # We compute the MSI from a SRF if necessary.
        msi = self.compute_msi(means, std) if self.band.size > 3 else self

        fcc = np.rot90(np.power(msi, gamma))[:, :, ::-1]

        ax_im.imshow(fcc)
        ax_im.set_title("False Color Composite")
        ax_im.set_xticks([])
        ax_im.set_yticks([])

        (line,) = ax_plot.plot(self.band.values, self.isel(x=0, y=0).values)
        ax_plot.set_title("Spectrum at pixel (0, 0)")
        ax_plot.set_xlabel("Wavelength [nm]")

        ax_plot.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
        ax_plot.set_xticklabels(
            [str(int(round(elem))) for elem in ax_plot.get_xticks()]
        )

        marker = ax_im.scatter([], [], s=100, color="chartreuse", marker="+")

        def update_plot(event):
            if event.inaxes == ax_im:
                x_click = int(event.xdata)
                y_click = int(event.ydata)

                # Take into account the rot90
                pixel_value = self.isel(
                    x=x_click, y=self.shape[1] - y_click
                ).values  # Not rotate unlike the FCC image, so no need to invert x_click and y_click
                line.set_ydata(pixel_value)
                ax_plot.set_ylim([0.0, max(0.035, pixel_value.max())])
                ax_plot.set_title(f"Spectrum at pixel ({x_click}, {y_click})")
                marker.set_offsets((x_click, y_click))
                plt.draw()

        _ = fig.canvas.mpl_connect("button_press_event", update_plot)
        fig.tight_layout()

        return fig

    @classmethod
    def from_spectrum(cls, data: np.ndarray, domain: SpectralDomain) -> "HSIDataArray":
        array = HSIDataArray(
            data=data,
            dims=["x", "y", "band"],
            coords=dict(
                x=np.arange(0, data.shape[0]),
                y=np.arange(0, data.shape[1]),
                band=np.linspace(domain.begin_nm, domain.end_nm, data.shape[2]),
            ),
            attrs=dict(band_cutoffs=[]),
        )
        return array

    @classmethod
    def from_numpy(
        cls, data: np.array, start_band: float, end_band: float
    ) -> "HSIDataArray":
        array = HSIDataArray(
            data=data,
            dims=["x", "y", "band"],
            coords=dict(
                x=np.arange(0, data.shape[0]),
                y=np.arange(0, data.shape[1]),
                band=np.linspace(start_band, end_band, data.shape[2]),
            ),
            attrs=dict(band_cutoffs=[]),
        )
        return array

    @classmethod
    def from_envi(cls, dataloader, region: Optional[tuple] = None):
        if region is not None:
            minx, maxx, miny, maxy = region
            data = dataloader.read_subregion((minx, maxx), (miny, maxy)).copy()
        else:
            data = dataloader.load().asarray().copy()

        array = HSIDataArray(
            data=data,
            dims=["x", "y", "band"],
            coords=dict(
                x=np.arange(0, data.shape[0]),
                y=np.arange(0, data.shape[1]),
                band=np.array(
                    [float(wav) for wav in dataloader.metadata["wavelength"]]
                ),
            ),
            attrs=dict(band_cutoffs=[]),
        )
        return array

    @property
    def start_band(self):
        return self.coords["band"][0].item()

    @property
    def end_band(self):
        return self.coords["band"][-1].item()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if isinstance(result, xr.DataArray):
            return HSIDataArray(
                data=result.data,
                dims=result.dims,
                coords=result.coords,
                attrs=result.attrs,
            )

        return result
