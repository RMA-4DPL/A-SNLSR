"""Generate reports on a model after training"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader

from a_snlsr.data import SpectralDomain
from a_snlsr.data.hsi import HSIDataArray
from a_snlsr.models.sisr import SISRNetwork


def save_metrics_tensorboard(model_folder: Path, report_folder: Path):
    sns.set_style(style="whitegrid")

    event_acc = EventAccumulator(model_folder.as_posix())
    event_acc.Reload()

    tags = event_acc.Tags()["scalars"]
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]

        fig, axis = plt.subplots(figsize=(6, 5))

        axis.plot(steps, values, label=tag)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Value")
        axis.legend()
        fig.savefig(report_folder / f"{tag.replace("/", "_")}.png")
        plt.close(fig)


def generate_figure_superres(
    input_hsi: list[HSIDataArray],
    output_hsi: list[HSIDataArray],
    target_hsi: list[HSIDataArray],
    input_msi: list[HSIDataArray],
    output_msi: list[HSIDataArray],
    target_msi: list[HSIDataArray],
    spectral_domains: Optional[list[SpectralDomain]] = None,
    domain_rel_indices: Optional[torch.Tensor] = None,
) -> matplotlib.figure.Figure:
    matplotlib.use("Agg")

    batch_size, height, width = (
        len(input_hsi),
        output_hsi[0].x.size,
        output_hsi[1].y.size,
    )

    # We compute the spectral angle mapper as error map
    sam_matrices = []

    for hr_hsi_instance, gt_hsi_instance in zip(output_hsi, target_hsi):
        dot_product = np.sum(hr_hsi_instance * gt_hsi_instance, axis=-1)
        norm_out = np.linalg.norm(hr_hsi_instance, axis=-1)
        norm_target = np.linalg.norm(gt_hsi_instance, axis=-1)
        sam = np.arccos(dot_product / (norm_out * norm_target))
        sam_matrices.append(sam)

    sam_matrices = np.array(sam_matrices)

    # We normalize the matrices to be plottable between 0 and 1
    min_sam = sam_matrices.min()
    max_sam = sam_matrices.max()
    sam_matrices = (sam_matrices - min_sam) / (max_sam - min_sam)

    # Setting up the figure
    fig = plt.figure(figsize=(15, 3 * batch_size))
    gs = gridspec.GridSpec(batch_size, 5, width_ratios=[1, 1, 1, 1, 2])

    # Generating the necessary axes.
    axes = []
    for row in range(batch_size):
        row_axes = []
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    axes = np.array(axes)
    markers = []
    markers_gt = []
    markers_lr = []
    lines = []
    lines_gt = []
    lines_lr = []
    lines_sam = []

    # We prepare the axes for the plot
    for instance_idx in range(batch_size):
        row_axes = axes[instance_idx]

        # We select a random point and we plot the curves in LR-HSI, HR-HSI and GT-HSI
        random_x, random_y = np.random.randint(0, height), np.random.randint(0, width)

        # We plot the LR-HSI
        row_axes[0].imshow(np.rot90(input_msi[instance_idx]))
        row_axes[0].grid(False)
        row_axes[0].set_xticks([])
        row_axes[0].set_yticks([])
        row_axes[0].set_title("LR-HSI", fontdict={"fontstyle": "italic"})

        if (spectral_domains is not None) and (domain_rel_indices is not None):
            domain = spectral_domains[domain_rel_indices[instance_idx]]
            row_axes[0].set_xlabel(f"Domain: {domain.domain_name}", fontsize=10)

        markers_lr.append(
            row_axes[0].scatter([], [], s=100, color="chartreuse", marker="+")
        )

        # We plot the HR-HSI
        row_axes[1].imshow(np.rot90(output_msi[instance_idx]))
        row_axes[1].grid(False)
        row_axes[1].set_xticks([])
        row_axes[1].set_yticks([])
        row_axes[1].set_title("HR-HSI (output)", fontdict={"fontstyle": "italic"})
        markers.append(
            row_axes[1].scatter(
                [random_x], [random_y], s=100, color="chartreuse", marker="+"
            )
        )

        # We plot the GT-HSI
        row_axes[2].imshow(np.rot90(target_msi[instance_idx]))
        row_axes[2].grid(False)
        row_axes[2].set_xticks([])
        row_axes[2].set_yticks([])
        row_axes[2].set_title(
            "HR-MSI & HR-HSI (target)", fontdict={"fontstyle": "italic"}
        )
        markers_gt.append(
            row_axes[2].scatter(
                [random_x], [random_y], s=100, color="chartreuse", marker="+"
            )
        )

        sam_error = sam_matrices[instance_idx]
        row_axes[3].imshow(np.rot90(sam_error), vmin=0.0, vmax=1.0, cmap="plasma")
        row_axes[3].grid(False)
        row_axes[3].set_xticks([])
        row_axes[3].set_yticks([])
        row_axes[3].set_title("Spectral Angle (↓)")
        lines_sam.append(
            row_axes[3].scatter(
                [random_x], [random_y], s=100, color="chartreuse", marker="+"
            )
        )

        (line,) = row_axes[4].plot(
            np.rot90(output_hsi[instance_idx])[random_x, random_y, :],
            color="b",
            label="Output",
        )
        lines.append(line)
        (line_gt,) = row_axes[4].plot(
            np.rot90(target_hsi[instance_idx])[random_x, random_y, :],
            color="r",
            label="GT",
        )
        lines_gt.append(line_gt)
        (line_lr,) = row_axes[4].plot(
            np.rot90(input_hsi[instance_idx])[random_x // 4, random_y // 4, :],
            color="g",
            label="LR",
        )
        lines_lr.append(line_lr)

        row_axes[4].set_title(f"Spectum at pixel ({random_x}, {random_y})")
        row_axes[4].set_xticks([])

        row_axes[4].legend()

    fig.tight_layout()

    return fig


def generate_report_sisr(
    model: SISRNetwork,
    model_folder: Path,
    val_dataloader: DataLoader,
    spectral_domain: SpectralDomain,
) -> None:
    report_folder = model_folder / "report"
    report_folder.mkdir(exist_ok=True)

    training_metrics_folder = report_folder / "training_metrics"
    training_metrics_folder.mkdir(exist_ok=True)

    # We open a batch of the dataset and perform inference
    batch = next(iter(val_dataloader))
    output = model.inference(batch)

    # We perform full validation of the model to compute statistics and write them.
    _, loss_dict = model.validate(val_dataloader, epoch=0)
    for k, v in loss_dict.items():
        loss_dict[k] = float(v)

    json.dump(loss_dict, (report_folder / "validation_metrics.json").open("w"))

    input_hsi = batch["lr_hsi"].cpu().numpy().transpose(0, 2, 3, 1)
    output_hsi = output.cpu().numpy().transpose(0, 2, 3, 1)
    target_hsi = batch["hr_hsi"].cpu().numpy().transpose(0, 2, 3, 1)

    # We create the figure
    input_hsi = [
        HSIDataArray.from_numpy(
            hsi,
            start_band=spectral_domain.begin_nm,
            end_band=spectral_domain.end_nm,
        )
        for hsi in input_hsi
    ]
    input_msi = [
        array.compute_msi(
            spectral_domain.msi_bands,
            spectral_domain.msi_band_width,
        )
        ** spectral_domain.msi_alpha_correction  # fmt: skip
        for array in input_hsi
    ]

    output_hsi = [
        HSIDataArray.from_numpy(
            hsi,
            start_band=spectral_domain.begin_nm,
            end_band=spectral_domain.end_nm,
        )
        for hsi in output_hsi
    ]

    output_msi = [
        array.compute_msi(
            spectral_domain.msi_bands,
            spectral_domain.msi_band_width,
        )
        ** spectral_domain.msi_alpha_correction  # fmt: skip
        for array in output_hsi
    ]

    target_hsi = [
        HSIDataArray.from_numpy(
            hsi,
            start_band=spectral_domain.begin_nm,
            end_band=spectral_domain.end_nm,
        )
        for hsi in target_hsi
    ]
    target_msi = [
        array.compute_msi(
            spectral_domain.msi_bands,
            spectral_domain.msi_band_width,
        )
        ** spectral_domain.msi_alpha_correction  # fmt: skip
        for array in target_hsi
    ]

    # Generate a figure with examples for super-resolution.
    figure = generate_figure_superres(
        input_hsi, output_hsi, target_hsi, input_msi, output_msi, target_msi
    )

    figure.savefig(report_folder / "validation_samples.png")
    plt.close(figure)

    # Generate a figure for every tensorboard metrics saved during training
    save_metrics_tensorboard(model_folder, training_metrics_folder)


def generate_report_adversarial_sr(
    model: SISRNetwork,
    model_folder: Path,
    val_dataloader: DataLoader,
    spectral_domains: list[SpectralDomain],
    gamma_correction: float = 0.4,
) -> None:
    report_folder = model_folder / "report"
    report_folder.mkdir(exist_ok=True)

    training_metrics_folder = report_folder / "training_metrics"
    training_metrics_folder.mkdir(exist_ok=True)

    # We open a batch of the dataset and perform inference
    batch = next(iter(val_dataloader))
    output = model.inference(batch)

    # We perform full validation of the model to compute statistics and write them.
    _, loss_dict = model.validate(val_dataloader, epoch=0)
    for k, v in loss_dict.items():
        loss_dict[k] = float(v)

    json.dump(loss_dict, (report_folder / "validation_metrics.json").open("w"))

    input_hsi = batch["lr_hsi"].cpu().numpy().transpose(0, 2, 3, 1)
    output_hsi = output.cpu().numpy().transpose(0, 2, 3, 1)
    target_hsi = batch["hr_hsi"].cpu().numpy().transpose(0, 2, 3, 1)

    domain_rel_indices = batch["domain_targets"].argmax(dim=1).cpu().numpy().astype(int)

    # We create the figure
    input_hsi = [
        HSIDataArray.from_numpy(
            hsi,
            start_band=spectral_domains[idx].begin_nm,
            end_band=spectral_domains[idx].end_nm,
        )
        for hsi, idx in zip(input_hsi, domain_rel_indices)
    ]
    input_msi = [
        array.compute_msi(
            spectral_domains[idx].msi_bands,
            spectral_domains[idx].msi_band_width,
        )
        ** spectral_domains[idx].msi_alpha_correction  # fmt: skip
        for array, idx in zip(input_hsi, domain_rel_indices)
    ]

    output_hsi = [
        HSIDataArray.from_numpy(
            hsi,
            start_band=spectral_domains[idx].begin_nm,
            end_band=spectral_domains[idx].end_nm,
        )
        for hsi, idx in zip(output_hsi, domain_rel_indices)
    ]

    output_msi = [
        array.compute_msi(
            spectral_domains[idx].msi_bands,
            spectral_domains[idx].msi_band_width,
        )
        ** spectral_domains[idx].msi_alpha_correction  # fmt: skip
        for array, idx in zip(output_hsi, domain_rel_indices)
    ]

    target_hsi = [
        HSIDataArray.from_numpy(
            hsi,
            start_band=spectral_domains[idx].begin_nm,
            end_band=spectral_domains[idx].end_nm,
        )
        for hsi, idx in zip(target_hsi, domain_rel_indices)
    ]
    target_msi = [
        array.compute_msi(
            spectral_domains[idx].msi_bands,
            spectral_domains[idx].msi_band_width,
        )
        ** spectral_domains[idx].msi_alpha_correction  # fmt: skip
        for array, idx in zip(target_hsi, domain_rel_indices)
    ]

    # Generate a figure with examples for super-resolution.
    figure = generate_figure_superres(
        input_hsi,
        output_hsi,
        target_hsi,
        input_msi,
        output_msi,
        target_msi,
        spectral_domains=spectral_domains,
        domain_rel_indices=domain_rel_indices,
    )

    figure.savefig(report_folder / "validation_samples.png")
    plt.close(figure)

    # Generate a figure for every tensorboard metrics saved during training
    save_metrics_tensorboard(model_folder, training_metrics_folder)
