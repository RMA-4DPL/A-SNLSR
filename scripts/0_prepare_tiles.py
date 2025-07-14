"""From the scenes we have, we cut the large tiles into subtiles"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as scio
from PIL import Image, ImageDraw
from spectral.io import envi
from tqdm import tqdm

from a_snlsr.data import SpectralDomain, SpectralDomainAction
from a_snlsr.data.hsi import HSIDataArray
from a_snlsr.logging import get_logger

logger = get_logger()


def find_hdr_files(
    input_folder: Path, file_suffix: str = ".img"
) -> list[tuple[Path, Path]]:
    """Recursively parses folders to find .HDR files and their matching .img file"""
    found_tiles = []
    for subpath in input_folder.iterdir():
        if subpath.is_dir():
            found_tiles.extend(find_hdr_files(subpath, file_suffix))
        elif subpath.suffix == ".hdr":
            if not subpath.with_suffix(file_suffix).exists:
                logger.warning(
                    f"Couldn't find matching .img file for .hdr file: {subpath.as_posix()}."
                )
                continue
            found_tiles.append((subpath, subpath.with_suffix(file_suffix)))

    return found_tiles


def draw_patches_bmp(
    bmp_path: Path, output_folder: Path, patches: list, tile_size: int = 128
) -> None:
    output_path = output_folder / (str(bmp_path.stem) + "_PATCHES.BMP")

    # Open the image file
    with Image.open(bmp_path) as img:
        # Create a draw object
        draw = ImageDraw.Draw(img)

        # Draw each patch as a green square
        for coords in patches:
            start_x, start_y = coords

            # Define the bounding box for the square
            end_x = start_x + tile_size
            end_y = start_y + tile_size

            # Draw the rectangle
            draw.rectangle(
                [start_x, start_y, end_x, end_y], outline=(0, 255, 0), width=3
            )

        # Save the modified image
        img.save(output_path)


def subdivide_patches(
    hdr_path: Path, img_path: Path, overlap_threshold: float = 0.5, tile_size: int = 128
) -> list:
    """Generates dictionnary to subdivise HSI scenes into smaller patches."""
    height, width = envi.open(hdr_path, img_path).shape[:2]
    selection_mask = np.zeros((height, width), dtype=bool)

    patches = []

    # Calculate the number of patches along width and height
    num_patches_height = np.ceil(height / tile_size).astype(int)
    num_patches_width = np.ceil(width / tile_size).astype(int)

    for i in range(num_patches_height):
        for j in range(num_patches_width):
            # Calculate the starting coordinates of the patch
            start_x = i * tile_size
            start_y = j * tile_size

            # Adjust for overlap at the edges
            if start_x + tile_size > height:
                start_x = height - tile_size
            if start_y + tile_size > width:
                start_y = width - tile_size

            # Check for the overlap
            zone_mask = selection_mask[
                start_x : start_x + tile_size, start_y : start_y + tile_size
            ]

            zone_size = zone_mask.shape[0] * zone_mask.shape[1]
            zone_percentage = zone_mask.astype(int).sum() / zone_size

            if zone_percentage < overlap_threshold:
                patches.append((start_x, start_y))
                selection_mask[
                    start_x : start_x + tile_size, start_y : start_y + tile_size
                ] = True

    return patches


def preprocess_hsi(
    hsi: HSIDataArray, output_bands: int, domain: SpectralDomain
) -> HSIDataArray:
    # Select first the spectral bands from the domain specifications.
    hsi = hsi.sel(band=slice(domain.begin_nm, domain.end_nm))

    # Reduce the bands of the HSI to a manageable amount of bands.
    hsi = hsi.spectral_downsample(output_bands)

    # We remove bands which are always uncovered by sunlight.
    for band_start, band_end in domain.cutoff_ranges:
        hsi = hsi.cutoff_bands(band_start, band_end)

    return hsi


def compute_dataset_statistics(
    scene_paths: list, output_bands: int, domain: SpectralDomain, training_names: list
) -> dict:
    band_min, band_max = [], []

    for hdr_path, img_path in tqdm(scene_paths, desc="Computing statistics over image"):
        if not any(substr in hdr_path.stem for substr in training_names):
            logger.info(
                f"Skipping tile {hdr_path.stem} as it is not considered for training."
            )
            continue

        # We load an HSI image from the envi dataloader
        image_loader = envi.open(hdr_path, img_path)
        hsi = HSIDataArray.from_envi(image_loader)

        hsi = preprocess_hsi(hsi, output_bands, domain)

        band_min.append(hsi.min(axis=(0, 1)).data)
        band_max.append(hsi.max(axis=(0, 1)).data)

    band_min = np.array(band_min).min(axis=0)
    band_max = np.array(band_max).max(axis=0)

    return {
        "band_min": band_min,
        "band_max": band_max,
    }


def process_patches(
    hdr_path: Path,
    img_path: Path,
    output_folder: Path,
    tiles: list,
    domain: SpectralDomain,
    output_bands: int,
    tile_size: int = 128,
    downsampling_factor: int = 4,
):
    dataspec = envi.open(hdr_path, img_path)

    psf_kernel = scio.loadmat(
        Path(__file__).parent.parent / "src/hyperres/data/blur_kernel.mat"
    )["data"]
    n_downsampling_steps = np.log2(downsampling_factor).astype(int)

    for idx, (start_x, start_y) in tqdm(enumerate(tiles), total=len(tiles)):
        hr_hsi = HSIDataArray.from_envi(
            dataspec, (start_x, start_x + tile_size, start_y, start_y + tile_size)
        )

        # Band reduction and band cutoff.
        hr_hsi = preprocess_hsi(hr_hsi, output_bands, domain)

        # Compute MSI
        hr_msi = hr_hsi.compute_msi(domain.msi_bands, domain.msi_band_width)

        # Compute the intermediate downsampled HSIs and the final input LR HSI.
        partial_downsample = {}
        lr_hsi, partial_downsampled_hsi = hr_hsi.spatial_downsample(
            downsampling_factor, progressive=True
        )

        for down_idx in range(n_downsampling_steps):
            if down_idx < n_downsampling_steps - 1:
                partial_downsample[f"lr_hsi_div{2 ** (down_idx + 1)}"] = (
                    partial_downsampled_hsi[down_idx].values
                )

        # Save patches as matlab matrix.
        output_file = output_folder / (hdr_path.stem + f"_N{idx}_SUBTILE.mat")
        data_output = {
            "hr_hsi": hr_hsi.values,
            "hr_msi": hr_msi.values,
            "lr_hsi": lr_hsi.values,
            "metadata": {
                "scene": hdr_path.stem,
                "tile_size": tile_size,
                "start_x": start_x,
                "start_y": start_y,
                "downsampling_factor": downsampling_factor,
                "psf_kernel": psf_kernel,
            },
        }
        data_output.update(partial_downsample)
        scio.savemat(output_file, data_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tiles preprocessing")
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Input folder where to find HDR files to process.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder where to write the processed cut tiles.",
    )
    parser.add_argument(
        "domain",
        action=SpectralDomainAction,
        choices=[domain.domain_name for domain in SpectralDomain],
        help="Which spectral domain is being processed.",
    )
    parser.add_argument(
        "--tile_size", type=int, default=128, help="Size of output for HSI tiles."
    )
    parser.add_argument(
        "--downsampling_factor",
        type=int,
        default=4,
        help="Size on which to downsample low-res HSI.",
    )
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=0.5,
        help="Maximum percentage of a tile that can be duplicate (usually at edges).",
    )
    parser.add_argument(
        "--input_bands",
        type=int,
        default=300,
        help="How many bands has the input data.",
    )
    parser.add_argument(
        "--output_bands",
        type=int,
        default=45,  # 45 for SWIR1, 38 for SWIR2 - in order to achieve 30 bands in total after cutoff of absorbed bands.
        help="How many bands should be derived in the first preprocessing steps. The final number of bands can still change due to SWIR band cutoffs.",
    )
    parser.add_argument(
        "--draw_bmp",
        action="store_true",
        help="If activated, a BMP preview of the scene with all patches subdivisions will be drawn.",
    )
    parser.add_argument(
        "--vnir",
        action="store_true",
        help="By default is SWIR. Enable if you want to process a VNIR dataset.",
    )

    args = parser.parse_args()

    if not args.output_folder.exists():
        logger.info(f"Creating folder: {args.output_folder.as_posix()}")
        args.output_folder.mkdir()

    logger.info(f"Finding HSI files in folder: {args.input_folder.as_posix()}...")
    scene_paths = find_hdr_files(args.input_folder, ".raw" if args.vnir else ".img")
    logger.info(f"Found {len(scene_paths)} scenes.")

    for hdr_path, img_path in scene_paths:
        logger.info(f"Subdiving rectangles for scene: {hdr_path.as_posix()}.")
        scene_rectangles = subdivide_patches(
            hdr_path,
            img_path,
            overlap_threshold=args.overlap_threshold,
            tile_size=args.tile_size,
        )
        logger.info(f"{len(scene_rectangles)} patches dervied from scene.")

        if (
            args.draw_bmp and hdr_path.with_suffix(".jpg" if args.vnir else ".BMP").exists()  # fmt: skip
        ):
            draw_patches_bmp(
                hdr_path.with_suffix(".jpg" if args.vnir else ".BMP"),
                args.output_folder,
                scene_rectangles,
                args.tile_size,
            )

        logger.info(f"Generate HSI and MSI patches for scene: {hdr_path.stem}...")
        process_patches(
            hdr_path,
            img_path,
            args.output_folder,
            scene_rectangles,
            domain=args.domain,
            output_bands=args.output_bands,
            tile_size=args.tile_size,
            downsampling_factor=args.downsampling_factor,
        )

    logger.info("Finished processing subpatches.")
