"""Utilities to plot."""

import matplotlib
import matplotlib.axis
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from a_snlsr.data.srf import LinearSRF


def visualize_srfs(
    srf: LinearSRF,
) -> tuple[matplotlib.figure.Figure, matplotlib.axis.Axis]:
    fig, axis = plt.subplots(figsize=(10, 5))

    color_map = [(0.0, "blue"), (0.5, "green"), (1.0, "red")]

    custom_cmap = LinearSegmentedColormap.from_list("blue_green_red", color_map)
    colors = [custom_cmap(i / srf.n_bands) for i in range(srf.n_bands)]

    x = np.linspace(srf.begin_nm, srf.end_nm, srf.spectral_steps)

    for i in range(srf.n_bands):
        axis.plot(x, srf.values[:, i], color=colors[i])

    axis.set_title("Spectral Response Function")
    axis.set_xlim((srf.begin_nm, srf.end_nm))
    axis.set_xlabel("Wavelength [nm]")
    axis.set_ylabel("Integration weight [AU]")

    fig.show()
    return fig, axis


def visualize_fcc_spectrum(
    hypercube: np.ndarray, srf: LinearSRF, gamma: float = 0.4, vnir: bool = False
):
    fig, (ax_image, ax_plot) = plt.subplots(ncols=2, figsize=(10, 3))

    fcc = (
        srf.apply(hypercube) ** gamma
    )  # Generate FCC from gaussian SRF with gamma correction

    if vnir:
        fcc = fcc[
            :, :, ::-1
        ]  # We want to show RGB order (instead of natural BGR order in the spectrum)

    ax_image.imshow(np.rot90(fcc))
    ax_image.set_title("FCC from SWIR image.")

    (line,) = ax_plot.plot(srf.spectral_centers, hypercube[0, 0, :])
    ax_plot.set_title("Spectrum at pixel (0, 0)")
    ax_plot.xaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
    ax_plot.set_xticklabels([str(int(round(elem))) for elem in ax_plot.get_xticks()])

    marker = ax_image.scatter([], [], s=100, color="chartreuse", marker="+")

    def update_plot(event):
        if event.inaxes == ax_image:
            x_click = int(event.xdata)
            y_click = int(event.ydata)

            pixel_value = hypercube[
                x_click, hypercube.shape[1] - y_click
            ]  # Not rotate unlike the FCC image, so no need to invert x_click and y_click
            line.set_ydata(pixel_value)
            ax_plot.set_ylim([0.0, max(0.035, pixel_value.max())])
            ax_plot.set_title(f"Spectrum at pixel ({x_click}, {y_click})")
            marker.set_offsets((x_click, y_click))
            plt.draw()

    _ = fig.canvas.mpl_connect("button_press_event", update_plot)

    fig.tight_layout()
    plt.show()
