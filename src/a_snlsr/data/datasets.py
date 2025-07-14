import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as scio
import torch
import torchvision.transforms.v2.functional as F2
from torch.utils.data import Dataset
from tqdm import tqdm

from a_snlsr.data import SpectralDomain


class TileDatasetSISR(Dataset):

    def __init__(
        self,
        dataset_folder: Path,
        selected_files: list,
        validation: bool,
        device: torch.device,
        augment: bool = False,
        blur_prob: float = 0.25,
        flip_prob: float = 0.5,
        rot_prob: float = 0.5,
        log_scale: bool = True,
    ):
        self.validation = validation
        self.dataset_folder = dataset_folder
        self.selected_files = selected_files
        self.device = device
        self.augment = augment
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.log_scale = log_scale

        self._read_dataset_stats()
        self._setup_data()

    def __len__(self):
        return self.n_tiles

    def __getitem__(self, item):
        data = {
            "hr_hsi": self.hr_hsi_data[item],  # .clone(),
            "lr_hsi": self.lr_hsi_data[item],  # .clone(),
            "hr_msi": self.hr_msi_data[item],  # .clone(),
        }
        part_gt = {}
        for key, elem in self.part_gt_data.items():
            part_gt[key] = elem[item].clone()

        if self.augment and not self.validation:
            data, part_gt = self._apply_augmentations(data, part_gt)

        for key, elem in part_gt.items():
            data[key] = elem
        return data

    def _preprocess_tile(self, X: torch.Tensor):
        """Preprocess a tensor with the computed statistics values"""
        if self.log_scale:
            X[X < 1e-4] = 1e-4
            X = torch.log(X)
        else:
            X[X > 5e-2] = 5e-2  # Clip values to avoid extreme values in the log scale
        return (X - self.stats["min_val"]) / (
            self.stats["max_val"] - self.stats["min_val"]
        )

    def _read_dataset_stats(self):
        """Read the dataset statistics on the selected files if it is a training dataset, or on other tiles than selected if it a validation dataset."""
        min_vals = []
        max_vals = []

        for subpath in tqdm(self.dataset_folder.iterdir()):
            if (subpath.suffix != ".mat") or (not subpath.is_file()):
                continue
            if ((subpath.stem in self.selected_files) and (not self.validation)) or (
                not (subpath.stem in self.selected_files) and self.validation
            ):
                # Read the tile data
                hsi_data = scio.loadmat(subpath)["hr_hsi"]

                # Log transformation of the data
                if self.log_scale:
                    hsi_data[hsi_data < 1e-4] = 1e-4
                    hsi_data = np.log(hsi_data)
                else:
                    hsi_data[hsi_data > 5e-2] = (
                        5e-2  # Clip values to avoid extreme values in the log scale
                    )

                min_vals.append(hsi_data.min())
                max_vals.append(hsi_data.max())

        self.stats = {
            "min_val": np.array(min_vals).min(),
            "max_val": np.array(max_vals).max(),
        }

    def _setup_data(self):
        self.lr_hsi_data = []
        self.hr_msi_data = []
        self.hr_hsi_data = []
        self.part_gt_data = defaultdict(list)

        for subpath in self.dataset_folder.iterdir():
            if (subpath.stem in self.selected_files) and (subpath.suffix == ".mat"):
                tile_data = scio.loadmat(subpath)
                self.lr_hsi_data.append(tile_data["lr_hsi"])
                self.hr_msi_data.append(tile_data["hr_msi"])
                self.hr_hsi_data.append(tile_data["hr_hsi"])

                partial_downsample_keys = [
                    key for key in tile_data.keys() if key.startswith("lr_hsi_div")
                ]
                for key in partial_downsample_keys:
                    self.part_gt_data[key].append(tile_data[key])
                self.part_gt_data["lr_hsi_div1"].append(tile_data["hr_hsi"])

        assert len(self.lr_hsi_data) == len(self.hr_msi_data)
        assert len(self.hr_msi_data) == len(self.hr_hsi_data)
        assert len(self.selected_files) == len(
            self.hr_hsi_data
        ), "Couldn't find all tiles in the selected files."

        self.n_tiles = len(self.selected_files)

        # Perform log-scaling of all the data
        self.lr_hsi_data = torch.from_numpy(np.stack(self.lr_hsi_data, axis=0).transpose(0, 3, 1, 2)).to(torch.float32).to(self.device)  # type: ignore
        self.hr_msi_data = torch.from_numpy(np.stack(self.hr_msi_data, axis=0).transpose(0, 3, 1, 2)).to(torch.float32).to(self.device)  # type: ignore
        self.hr_hsi_data = torch.from_numpy(np.stack(self.hr_hsi_data, axis=0).transpose(0, 3, 1, 2)).to(torch.float32).to(self.device)  # type: ignore

        self.lr_hsi_data = self._preprocess_tile(self.lr_hsi_data)
        self.hr_msi_data = self._preprocess_tile(self.hr_msi_data)
        self.hr_hsi_data = self._preprocess_tile(self.hr_hsi_data)

        for key, elem in self.part_gt_data.items():
            self.part_gt_data[key] = torch.from_numpy(np.stack(elem, axis=0).transpose(0, 3, 1, 2)).to(torch.float32).to(self.device)  # type: ignore

        self.size = [
            self.hr_hsi_data[0].shape[0],
            self.hr_hsi_data[0].shape[1],
            self.hr_hsi_data[0].shape[2],
            self.lr_hsi_data[0].shape[0],
            self.lr_hsi_data[0].shape[1],
            self.hr_msi_data[0].shape[2],
        ]

    # Apply augmentations to the data
    # Creates an additional "lr_hsi_clean" key in the data dictionary
    # which contains the originallr_hsi with only non-training augmentations applied.
    def _apply_augmentations(self, data, part_gt):
        # Random horizontal flip
        if random.random() < self.flip_prob:
            for k in ["lr_hsi", "hr_hsi", "hr_msi"]:
                data[k] = F2.horizontal_flip(data[k])
            for k in ["lr_hsi_div2", "lr_hsi_div1"]:
                part_gt[k] = F2.horizontal_flip(part_gt[k])

        # Random vertical flip
        if random.random() < self.flip_prob:
            for k in ["lr_hsi", "hr_hsi", "hr_msi"]:
                data[k] = F2.vertical_flip(data[k])
            for k in ["lr_hsi_div2", "lr_hsi_div1"]:
                part_gt[k] = F2.vertical_flip(part_gt[k])

        data["lr_hsi_clean"] = data["lr_hsi"].clone()

        # Gaussian blur (only on lr_hsi)
        if random.random() < self.blur_prob:
            data["lr_hsi"] = F2.gaussian_blur(data["lr_hsi"], kernel_size=3)

        return data, part_gt


class TileDatasetAbundance(Dataset):
    """Dataset contianing HSI tiles and their specral domain.
    This dataset can be used for domain adapatative abundance estimators.
    """

    def __init__(
        self,
        dataset_folders: list[Path],
        dataset_domains: list[SpectralDomain],
        selected_files: dict,
        validation: bool,
        device: torch.device,
        statistics: Optional[dict] = None,
        log_scale: bool = True,
    ):
        self.dataset_folders = dataset_folders
        self.dataset_domains = dataset_domains
        self.selected_files = selected_files
        self.validation = validation
        self.device = device
        self.log_scale = log_scale

        if statistics is None:
            self._read_dataset_stats()
        else:
            self.stats = statistics

        self._setup_data()

    @classmethod
    def search_mats(cls, path: Path):
        if not path.is_dir() and path.suffix == ".mat":
            return [path]

        found_matrices = []
        for subpath in path.iterdir():
            if subpath.is_dir():
                found_matrices.extend(TileDatasetAbundance.search_mats(subpath))
            elif subpath.suffix == ".mat":
                found_matrices.append(subpath)

        return found_matrices

    def __len__(self):
        return self.n_tiles

    def _read_dataset_stats(self):
        """Read the dataset statistics on the selected files if it is a training dataset, or on other tiles than selected if it a validation dataset."""
        stats = {}

        # Statistics are computed per domain, as we want to have both domains in the same data range.
        for idx, domain in enumerate(self.dataset_domains):
            # We look for the tiles in the current domain
            tile_paths = TileDatasetAbundance.search_mats(self.dataset_folders[idx])

            stats[domain] = {}

            min_vals = []
            max_vals = []

            for subpath in tqdm(
                tile_paths,
                desc=f"Computing dataset statistics for domain {domain.domain_name}",
            ):
                dom_selected_files = self.selected_files[domain.domain_name]
                if ((subpath.stem in dom_selected_files) and (not self.validation)) or (
                    not (subpath.stem in dom_selected_files) and self.validation
                ):
                    # Read the tile data
                    hsi_data = scio.loadmat(subpath)["hr_hsi"]

                    # Log transformation of the data
                    if self.log_scale:
                        hsi_data[hsi_data < 1e-4] = 1e-4
                        hsi_data = np.log(hsi_data)
                    else:
                        hsi_data[hsi_data > 5e-2] = (
                            5e-2  # Clip values to avoid extreme values in the log scale
                        )

                    min_vals.append(hsi_data.min())
                    max_vals.append(hsi_data.max())

            stats[domain] = {
                "min_val": np.array(min_vals).min(),
                "max_val": np.array(max_vals).max(),
            }

        self.stats = stats

    def _setup_data(self):
        self.hr_hsi_data = []
        self.lr_hsi_data = []
        self.domain_indices = []

        if len(self.dataset_folders) != len(self.dataset_domains):
            raise ValueError(
                "Both `dataset_folders` and `dataset_domains` should have the same size."
            )

        for folder, domain in zip(self.dataset_folders, self.dataset_domains):
            domain_paths = TileDatasetAbundance.search_mats(folder)

            selected_files = self.selected_files[domain.domain_name]

            for path in tqdm(
                domain_paths, desc=f"Loading tiles for domain {domain.domain_name}"
            ):
                if path.stem not in selected_files:
                    continue  # We ignore files that are not selected

                # Read the high res and low res HSIs
                tile_data = scio.loadmat(path)
                self.hr_hsi_data.append(
                    self._preprocess_tiles(tile_data["hr_hsi"], domain)
                )
                self.lr_hsi_data.append(
                    self._preprocess_tiles(tile_data["lr_hsi"], domain)
                )
                self.domain_indices.append(domain.id)

        self.n_tiles = len(self.hr_hsi_data)
        self.hr_hsi_data = torch.from_numpy(np.stack(self.hr_hsi_data, axis=0).transpose(0, 3, 1, 2)).to(torch.float32).to(self.device)  # type: ignore
        self.lr_hsi_data = torch.from_numpy(np.stack(self.lr_hsi_data, axis=0).transpose(0, 3, 1, 2)).to(torch.float32).to(self.device)  # type: ignore

        self.domain_indices = np.array(self.domain_indices)
        domain_targets = np.zeros(
            (self.domain_indices.shape[0], len(self.dataset_domains))
        )
        for domain_id_col, domain in enumerate(self.dataset_domains):
            domain_targets[self.domain_indices == domain.id, domain_id_col] = 1

        self.domain_targets = torch.from_numpy(domain_targets.astype(np.float32)).to(
            self.device
        )
        self.domain_indices = torch.from_numpy(
            self.domain_indices.astype(np.float32)
        ).to(self.device)

    def _preprocess_tiles(self, X: np.ndarray, domain: SpectralDomain):
        """Preprocess a tensor with the computed statistics values"""
        if self.log_scale:
            X[X < 1e-4] = 1e-4
            X = np.log(X)
        else:
            X[X > 5e-2] = 5e-2  # Clip values to avoid extreme values in the log scale

        return (X - self.stats[domain]["min_val"]) / (
            self.stats[domain]["max_val"] - self.stats[domain]["min_val"]
        )

    def __getitem__(self, item):
        data = {
            "hr_hsi": self.hr_hsi_data[item],
            "lr_hsi": self.lr_hsi_data[item],
            "domain": self.domain_indices[item],
            "domain_targets": self.domain_targets[item],
        }
        return data
