"""Logging utilities for TechBioT."""

import logging
from typing import Optional


# We define a custom logger class to add new methods.
class SuperResLogger(logging.Logger):
    def layer_debug(self, message, *args, **kwargs):
        if self.isEnabledFor(LAYER_DEBUG):
            self._log(LAYER_DEBUG, message, args, **kwargs)


# We setup that class as default for initialization
logging.setLoggerClass(SuperResLogger)

# Define my custom logger object
logger: Optional[SuperResLogger] = None  # type: ignore


COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
}
RESET_COLOR = "\033[0m"

# Defining a custom logging level for the input/output shapes of network layers
LAYER_DEBUG = 14

logging.addLevelName(LAYER_DEBUG, "LayerDebug")


class ColorFormatter(logging.Formatter):
    """Custom color formatter that adds colors to log levels."""

    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET_COLOR)
        message = super().format(record)
        return f"{log_color}{message}{RESET_COLOR}"


def get_logger() -> SuperResLogger:
    global logger

    if logger is None:
        logger = logging.getLogger("HyperRes")  # type: ignore
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # formatter = ColorFormatter("[%(asctime)s - %(name)s]: %(levelname)s - %(message)s")
        formatter = ColorFormatter("[%(asctime)s - %(levelname)s]: %(message)s")
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger
