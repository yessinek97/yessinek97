"""Module used to define all the helper functions for torch model."""
from logging import Logger

import torch

from ig.src.logger import get_logger

log: Logger = get_logger("utils/torch")


def get_device() -> str:
    """Function to check if CUDA is available and return the appropriate device.

    Returns a string indicating the selected device.
    """
    if torch.cuda.is_available():
        device = "cuda"
        log.info("CUDA is available. Using GPU.")
    else:
        device = "cpu"
        log.info("CUDA is not available. Using CPU.")
    return device
