"""Training functions for the fusion techniques"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from a_snlsr.logging import get_logger
from a_snlsr.models import SerializableNetwork
from a_snlsr.utils.misc import launch_tensorboard_process

logger = get_logger()


def get_last_epoch(tfevent_file: Path, key="Train/Loss"):
    tfevent_file_str = str(tfevent_file)
    ea = event_accumulator.EventAccumulator(tfevent_file_str)
    ea.Reload()

    if key not in ea.scalars.Keys():
        raise ValueError(
            "Couldn't find the value: '{key}' in tfevents, cannot evaluate which epoch to resume."
        )

    scalar_events = ea.scalars.Items(key)

    last_event = scalar_events[-1]

    return last_event.step


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch=-1,
):
    def lr_lambda(current_step):
        learning_rate = max(
            0.0, 1.0 - (float(current_step) / float(num_training_steps))
        )
        learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
        return learning_rate

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    final_lr: float,
    scheduler_patience: Optional[int] = 30,
    scheduler_factor: Optional[float] = 0.5,
):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_epochs, eta_min=final_lr
        )
    elif scheduler_name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=num_epochs // 4, T_mult=1, eta_min=final_lr
        )
    elif scheduler_name == "linear_warmup":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_epochs
        )
    elif scheduler_name == "plateau":
        if scheduler_patience is None or scheduler_factor is None:
            raise ValueError(
                "Scheduler patience and factor must be provided for plateau scheduler."
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )
    else:
        raise ValueError(
            f"Unknown scheduler name: {scheduler_name}. Available options: 'cosine', 'cosine_warm_restarts', 'linear_warmup', 'plateau'."
        )


def is_metric_better(
    metric: float, best_metric: float, metric_name: str = "Loss"
) -> bool:
    if "PSNR" in metric_name or "SSIM" in metric_name or "Accuracy" in metric_name:
        # For metrics like PSNR, SSIM, and Accuracy, higher is better
        return metric > best_metric
    else:
        # For metrics like Loss, lower is better
        return metric < best_metric


def train_model(
    model: SerializableNetwork,
    device: torch.device,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    save_path: Path,
    save_every_epoch: bool = False,
    num_epochs: int = 100,
    scheduler_name: str = "cosine_warm_restarts",
    learning_rate: float = 1e-3,
    final_lr: float = 1e-4,
    scheduler_metric: str = "Val/Loss",
    scheduler_patience: int = 30,
    scheduler_factor: float = 0.5,
    resume: bool = False,
    transfer_learning: Optional[Path] = None,
    tensorboard: bool = False,
) -> SerializableNetwork:
    """Perform the training on a fresh auto-encoder or classifier `model` with the given `dataloader`.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model to train which is an auto-encoder (outputs are expected to be in the same shape and form as the input).
    dataloader: DataLoader
        A dataloader yielding only input data (target data in unexistent)
    val_dataloader: Optional[DataLoader]
        A dataloader containing validation data. Will be evaluated at the end of each epoch.
    save_path: Optional[Path]
        Path to a folder where to save the model and it's epochs.

    Returns
    -------
    model: nn.Module
        The trained model.
    """

    if resume and transfer_learning is not None:
        raise ValueError("Cannot resume and transfer learning at the same time.")

    # Pushes the model to the GPU if it is available
    model = model.to(device)

    # Creating a directory where to save model weights, inference results and tensorboard records.
    if not save_path.exists():
        logger.info("Creating empty directory: %s", str(save_path))
        save_path.mkdir(parents=True)
    elif not save_path.is_dir():
        raise ValueError(f"Provided save_path {str(save_path)} exists and is a file.")

    # Load back the model weights
    if resume or transfer_learning:
        final_weights = (
            save_path / "model_final.pth"
            if not transfer_learning
            else transfer_learning
        )
        if not final_weights.exists():
            raise ValueError(
                "Cannot resume model training: %s doesn't exists.", str(final_weights)
            )
        logger.info("Re-load previous model weights...")
        model = SerializableNetwork.load(final_weights).to(device)

    # Initializes the loss function and the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        scheduler_name,
        optimizer,
        num_epochs,
        final_lr,
        scheduler_patience,
        scheduler_factor,
    )

    start_epoch = 0

    # Initializes the tensoboard recorder.
    if save_path is not None:
        if resume and not transfer_learning:
            event_files = [
                f for f in save_path.iterdir() if "events.out.tfevents" in f.stem
            ]
            if len(event_files) > 0:
                latest_event_file = list(sorted(event_files))[-1]
                start_epoch = get_last_epoch(latest_event_file)
                logger.info("Resuming from epoch: %s.", start_epoch)

        writer = SummaryWriter(log_dir=save_path)

    else:
        temp_dir = TemporaryDirectory()
        logger.warning(
            "No save_path was provided, saving tensorboard record on temorary directory: %s",
            str(temp_dir),
        )
        writer = SummaryWriter(log_dir=temp_dir)

    if tensorboard:
        launch_tensorboard_process(save_path)

    best_metric = (
        -np.inf
        if "PSNR" in scheduler_metric
        or "SSIM" in scheduler_metric
        or "Accuracy" in scheduler_metric
        else np.inf
    )

    try:
        for epoch in range(start_epoch, num_epochs):

            # Runs throught the dataloader once
            logger.info("Running training...")
            epoch_stats = model.train_epoch(optimizer, dataloader, epoch)

            log_message = f"Epoch {epoch + 1}/{num_epochs} "
            for stat_name, value in epoch_stats.items():
                writer.add_scalar(stat_name, value, epoch)
                log_message += f"| {stat_name} = {value} "

            logger.info(log_message)

            # Save the model weights on the current epoch
            if (save_path is not None) and save_every_epoch:
                model_save_path = save_path / f"model_epoch_{epoch + 1}.pth"
                model.save(model_save_path)
                logger.info("Model weights saved at: %s", str(save_path))

            # Runs the model on the validation dataset and assesses validation performance.
            if val_dataloader is not None:
                logger.info("Running validation...")
                _, results = model.validate(val_dataloader, epoch)

                log_message = f"Epoch {epoch + 1}/{num_epochs}"
                for name, value in results.items():
                    writer.add_scalar(name, value, epoch)
                    log_message += f" | {name} = {value}"

                logger.info("Validation: " + log_message)

                if scheduler_metric not in results:
                    raise ValueError(
                        f"Scheduler metric '{scheduler_metric}' not found in validation metrics. Available metrics: {list(results.keys())}"
                    )

                # Saves the model if the validation loss was improved.
                if is_metric_better(
                    results[scheduler_metric], best_metric, scheduler_metric
                ):
                    best_metric = results[scheduler_metric]
                    model_save_path = save_path / "model_best.pth"
                    logger.info(
                        "Best model reached so far, saving it to path: %s",
                        model_save_path,
                    )
                    model.save(model_save_path)

            # Updates the learning rate
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # If the scheduler is a plateau scheduler, it needs to be stepped with the validation loss.
                scheduler.step(results[scheduler_metric])
            else:
                # Otherwise, it is stepped with the epoch number.
                scheduler.step()

            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            logger.info("LR value set to %s", scheduler.get_last_lr()[0])

    except KeyboardInterrupt:
        logger.error("Training interrupted...")

    logger.info("Model trained sucessfully.")
    model.eval()
    writer.close()

    # Save the model weights as they are after the end of the training
    if save_path is not None:
        logger.info("Saving the trained model: %s", str(save_path))
        model_save_path = save_path / "model_final.pth"
        model.save(model_save_path)

    # Load back the model from the weights that performed the best on the validation loss.
    logger.info("Loading the model weights by the best weights so far")
    if best_metric < np.inf:
        model_save_path = save_path / "model_best.pth"
        if not model_save_path.exists():
            logger.error("Couldn't load the model by its best weights.")
            return model
        # Load the model weights and applies them to the model.
        model_weights = torch.load(model_save_path, map_location=device)
        model.load_state_dict(model_weights["model_state_dict"])

    return model
