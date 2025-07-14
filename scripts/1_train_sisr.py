"""Training of super-resolution algorithm..."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from a_snlsr.data import SpectralDomain, SpectralDomainAction
from a_snlsr.data.datasets import TileDatasetSISR
from a_snlsr.logging import get_logger
from a_snlsr.models import SerializableNetwork
from a_snlsr.models.reports import generate_report_sisr
from a_snlsr.models.sisr import (  # noqa
    SNLSR,
    ESRTNetwork,
    RCAMSRNetwork,
    SRFormerNetwork,
)
from a_snlsr.models.train import train_model
from a_snlsr.utils.device import load_device
from a_snlsr.utils.misc import list_tile_names

logger = get_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fusion Network training.")
    parser.add_argument(
        "input_folder", type=Path, help="Path to the folder with training tiles."
    )
    parser.add_argument(
        "domain",
        action=SpectralDomainAction,
        choices=[domain.domain_name for domain in SpectralDomain],
        help="Which spectral domain is processed.",
    )
    parser.add_argument(
        "--split_names",
        type=Path,
        default=None,
        help="Path to the input tiles to split in train/val.",
    )
    parser.add_argument(
        "--model_folder",
        type=Path,
        required=False,
        help="To save the trained model on a specific path.",
    )
    parser.add_argument(
        "--hyperparams", type=Path, required=False, help="Path to a .pth file"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of data instances in every training iteration.",
    )
    parser.add_argument(
        "--base_lr", type=float, default=1e-3, help="Starting learning rate."
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=1e-5,
        help="Final learning rate, used if decaying learning rates are used.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Scheduler patience when reaching a pleateau",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Scheduler factor of reducing/augmenting the LR whenever a plateau is reached.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Whether to use tensorboard for logging.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to use deterministic training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2049,
        help="Random seed for reproducibility. Only used if --deterministic is set.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Wherever to resume a training."
    )
    parser.add_argument(
        "--transfer_learning",
        type=Path,
        default=None,
        help="Path to a model to transfer learning from.",
    )

    args = parser.parse_args()
    model_folder = args.model_folder

    logger.info("Find device for training...")
    device = load_device()

    if args.deterministic:
        logger.info("Setting up deterministic training...")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
        model = SRFormerNetwork(
            hsi_bands=30,
            hsi_width=32,
            hsi_height=32,
            super_resolution_factor=4,
            bicubic_skip=True,
        )
        # model = ESRTNetwork(
        #     hsi_bands=30,
        #     hsi_width=32,
        #     hsi_height=32,
        #     super_resolution_factor=4,
        #     bicubic_skip=False,
        # )
        # model = RCAMSRNetwork(
        #     hsi_bands=30,
        #     hsi_width=32,
        #     hsi_height=32,
        #     upsample_steps=2,
        #     n_features=64,
        #     n_dense=4,
        #     n_rcam=4,
        #     bicubic_skip=True,
        # )
        # model = SNLSR(
        #     hsi_bands=30,
        #     hsi_width=32,
        #     hsi_height=32,
        #     n_features=64,
        #     n_materials=16,
        #     super_resolution_factor=4,
        #     weight_spectral_loss=0.1,
        # )
        model.save_hyperparams(model_folder / "hparams.json")
        logger.info("Initializing default model: %s.", model)

    if not args.input_folder.is_dir():
        raise ValueError(
            "Input folder should be a folder, either is unexisting or is a file."
        )

    # Initializing train/val splits from tile names or existing split file.
    logger.info("Listing input tiles from folder: %s", args.input_folder)
    tile_names = list_tile_names(args.input_folder)

    if args.split_names is not None:
        logger.info("Reading tile splits from file: %s...", args.split_names)
        splits = np.load(args.split_names, allow_pickle=True)
        train_names, val_names = splits["train"], splits["val"]

    else:
        logger.info(
            "No split file specified, generating splits manually from parsed tile names."
        )
        train_names, val_names = train_test_split(
            tile_names, test_size=0.25, random_state=2049
        )

    logger.info("Saving tile splits in model folder.")
    np.savez(model_folder / "splits.npz", train=train_names, val=val_names)

    logger.info("Initializing datasets and dataloaders...")
    train_dataset = TileDatasetSISR(
        args.input_folder,
        train_names,
        validation=False,
        augment=False,
        blur_prob=0.0,
        rot_prob=0.0,
        flip_prob=0.0,
        device=device,
    )
    val_dataset = TileDatasetSISR(
        args.input_folder, val_names, validation=True, device=device
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    train_model(
        model=model,
        device=device,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_path=model_folder,
        save_every_epoch=False,
        num_epochs=args.n_epochs,
        learning_rate=args.base_lr,
        final_lr=args.final_lr,
        scheduler_factor=args.lr_factor,
        scheduler_patience=args.patience,
        resume=args.resume,
        transfer_learning=args.transfer_learning,
        tensorboard=args.tensorboard,
    )

    logger.info("Model training finished, generating reports...")

    # Increase the batch size of the val dataloader to 8 for the report generation

    # Fixed shuffle seed, no matter the seed selected for training
    np.random.seed(2049)
    torch.manual_seed(2049)
    torch.cuda.manual_seed(2049)

    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    generate_report_sisr(
        model=model,  # type: ignore
        model_folder=model_folder,
        val_dataloader=val_dataloader,
        spectral_domain=args.domain,
    )
