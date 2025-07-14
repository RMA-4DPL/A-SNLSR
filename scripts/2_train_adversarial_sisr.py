"""
Training procedure for models transforming material abundance maps to high-resolution hyperspectral images (HSI).
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from a_snlsr.data import SpectralDomain, SpectralDomainAction
from a_snlsr.data.datasets import TileDatasetAbundance
from a_snlsr.logging import get_logger
from a_snlsr.models import SerializableNetwork
from a_snlsr.models.reports import generate_report_adversarial_sr
from a_snlsr.models.sisr import AdversarialSNLSR
from a_snlsr.models.train import train_model
from a_snlsr.utils.device import load_device

logger = get_logger()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Material Super-Resolution Network training.")
    parser.add_argument(
        "--input_folders",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to the folders with training tiles.",
    )
    parser.add_argument(
        "--domains",
        required=True,
        action=SpectralDomainAction,
        choices=[domain.domain_name for domain in SpectralDomain],
        nargs="+",
        help="Specrtal domain ID (for example VNIR=0 and SWIR=1). Provide one for each input folder.",
    )
    parser.add_argument(
        "--split_names",
        type=str,
        required=True,
        help="Path to the input tiles to split in train/val.",
    )
    parser.add_argument(
        "--n_materials",
        type=int,
        default=16,
        help="Number of materials in the abundance map.",
    )
    parser.add_argument(
        "--discriminator_loss_ratio",
        type=float,
        default=0.01,
        help="Ratio of the discriminator loss to the total loss.",
    )
    parser.add_argument(
        "--model_folder",
        type=Path,
        required=False,
        help="Path where to save the trained model.",
    )
    parser.add_argument(
        "--hyperparams",
        type=Path,
        required=False,
        help="Path to a .pth file with hyperparameters for the model.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_warm_restarts",
        choices=["cosine_warm_restarts", "plateau", "cosine", "linear_warmup"],
        help="Type of learning rate scheduler to use.",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=1e-3,
        help="Base learning rate for the optimizer.",
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=1e-4,
        help="Final learning rate, used if decaying learning rates are used.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Patience for the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Factor by which to reduce the learning rate when a plateau is reached.",
    )
    parser.add_argument(
        "--bicubic_skip",
        action="store_true",
        help="Whether to use bicubic skip connections in the model.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Whether to run TensorBoard during training.",
    )

    args = parser.parse_args()

    model_folder = args.model_folder

    logger.info("Find device for training...")
    device = load_device()

    # Setting up model folder, either from new intialized folder or from exisiting folder
    if not model_folder:
        model_folder = Path(__file__).parent.parent / "models" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # fmt: skip
        logger.info("Model folder not specified, initialzed at: %s", model_folder)
    else:
        model_folder = Path(str(args.model_folder))

    if not model_folder.exists():
        logger.info("Creating model folder at %s", model_folder)
        model_folder.mkdir(parents=True)

    # Intializing the model, either from a serialized model in the model folder, either from specified hyperparameters or either from default parameters
    if (model_folder / "model_final.pth").exists():
        model = SerializableNetwork.load(model_folder / "model_final.pth")
        logger.info("Loaded model from serialization: %s.", model)
    elif args.hyperparams is not None:
        model = SerializableNetwork.initialize_json(args.hyperparams)
        model.save_hyperparams(model_folder / "hparams.json")
        logger.info("Initialized model from hyperparameters: %s.", model)
    else:
        model = AdversarialSNLSR(
            hsi_bands=30,
            hsi_width=32,
            hsi_height=32,
            n_features=64,
            n_materials=args.n_materials,
            super_resolution_factor=4,
            weight_spectral_loss=0.1,
            discriminator_loss_ratio=args.discriminator_loss_ratio,
            val_dataset_statistics=None,
            n_domains=len(args.domains),  # Number of domains for the discriminator
            bicubic_skip=args.bicubic_skip,
        )
        model.save_hyperparams(model_folder / "hparams.json")
        logger.info("Initializing default model: %s.", model)

    # Initializing train/val splits from tile names or existing split file.
    tile_names = []
    for input_folder in args.input_folders:
        tile_names.extend(
            [path.stem for path in TileDatasetAbundance.search_mats(input_folder)]
        )

    if args.split_names is not None:
        logger.info("Reading tile splits from file: %s...", args.split_names)
        splits = np.load(args.split_names, allow_pickle=True)
        train_names, val_names = splits["train"].item(), splits["val"].item()
    else:
        logger.info(
            "No split file specified, generating splits manually from parsed tile names."
        )
        train_names, val_names = train_test_split(
            tile_names, test_size=0.25, random_state=2049
        )

    logger.info("Saving tile splits in model folder.")
    np.savez(model_folder / "splits.npz", train=train_names, val=val_names)

    # ATTENTION: STATISTICS COMPUTED OVER SWIR1 AND APPLIED TO BOTH SWIR1 AND SWIR2
    stats = {
        SpectralDomain.SWIR_1: {
            "min_val": -9.2103405,
            "max_val": 0.6931473,
        },
        SpectralDomain.SWIR_2: {
            "min_val": -9.2103405,
            "max_val": 0.6931473,
        },
    }

    logger.info("Initializing datasets and dataloaders...")
    train_dataset = TileDatasetAbundance(
        args.input_folders,
        args.domains,
        train_names,
        validation=False,
        device=device,
        statistics=stats,
    )
    val_dataset = TileDatasetAbundance(
        args.input_folders,
        args.domains,
        val_names,
        validation=True,
        device=device,
        statistics=stats,
    )

    torch.manual_seed(2049)
    np.random.seed(2049)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    train_model(
        model=model,
        device=device,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_path=model_folder,
        save_every_epoch=False,
        num_epochs=args.n_epochs,
        scheduler_name=args.scheduler,
        learning_rate=args.base_lr,
        final_lr=args.final_lr,
        scheduler_metric="Val/SSIM",
        scheduler_factor=args.lr_factor,
        scheduler_patience=args.patience,
        tensorboard=args.tensorboard,
    )

    logger.info("Model training finished, generating reports...")

    # Increase the batch size of the val dataloader to 8 for the report generation

    # Fixed shuffle seed, no matter the seed selected for training
    np.random.seed(2049)
    torch.manual_seed(2049)
    torch.cuda.manual_seed(2049)

    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    generate_report_adversarial_sr(
        model=model,  # type: ignore
        model_folder=model_folder,
        val_dataloader=val_dataloader,
        spectral_domains=args.domains,
        gamma_correction=1.0,
    )
