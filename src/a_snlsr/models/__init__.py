"""Models for MSI-guided spatial super-resolution of HSI based on fusion."""

import json
from abc import abstractmethod
from functools import cache
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class SerializableNetwork(nn.Module):

    def __init__(self):
        super(SerializableNetwork, self).__init__()
        self.h_params = {}

    @property
    @cache
    def _device(self):
        return next(self.parameters(), self.buffers()).device  # type: ignore

    @abstractmethod
    def train_epoch(
        self, optimizer: Optimizer, dataloader: DataLoader, epoch: int = 0
    ) -> dict:
        pass

    @abstractmethod
    def validate(self, dataloader: DataLoader, epoch: int = 0) -> tuple[float, dict]:
        pass

    def save(self, path: Path) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "h_params": self.h_params,
                "model_class": self.__class__.__name__,
            },
            path,
        )

    def save_hyperparams(self, path: Path) -> None:
        json.dump(
            {"h_params": self.h_params, "model_class": self.__class__.__name__},
            path.open("w"),
        )

    @staticmethod
    def load(path: Path) -> "SerializableNetwork":
        # Importing the models that can be loaded
        from a_snlsr.models.sisr import (  # noqa isort:skip
            SNLSR,
            AdversarialSNLSR,
            ESRTNetwork,
            RCAMSRNetwork,
            SRFormerNetwork,
        )

        checkpoint = torch.load(path)
        model_class = locals()[checkpoint["model_class"]]
        model = model_class(**checkpoint["h_params"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def initialize_json(path: Path) -> "SerializableNetwork":
        checkpoint = json.load(path.open("r"))
        model_class = globals()[checkpoint["model_class"]]
        model = model_class(**checkpoint["h_params"])
        return model

    def __repr__(self):
        return f"{self.__class__.__name__}({{ {self.h_params} }})"
