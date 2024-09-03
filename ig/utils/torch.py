"""Module used to define all the helper functions for torch model."""
import gc
import random
from logging import Logger
from typing import List, Union

import torch
import torch.nn.functional as functional
from multimolecule import RiNALMoModel, RnaTokenizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torcheval.metrics.functional import binary_auroc, binary_f1_score
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from ig.utils.logger import get_logger

log: Logger = get_logger("utils/torch")

TOKENIZER_SOURCES = {
    "RnaTokenizer": RnaTokenizer,
    "AutoTokenizer": AutoTokenizer,
}
LLM_MODEL_SOURCES = {
    "AutoModel": AutoModel,
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
    "RiNALMoModel": RiNALMoModel,
}


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "sum",
) -> torch.Tensor:
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.

    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n \
            Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return loss


def find_mut_token(wild_type: torch.Tensor, mutated: torch.Tensor) -> Union[int, None]:
    """Returns the position of the token containing the mutation."""
    for tok_idx, _ in enumerate(wild_type):
        if wild_type[tok_idx] != mutated[tok_idx]:
            return tok_idx
    raise ValueError("Both sequences are identical")


def create_scheduler(
    scheduler_config: dict, num_epochs: int, steps_per_epoch: int, optimizer: Optimizer
) -> LRScheduler:
    """Returns lr scheduler according to config.

    Args:
        scheduler_config (dict): _description_

    Returns:
        CyclicLR: _description_
    """
    scheduler = OneCycleLR(
        optimizer,
        max_lr=scheduler_config["max_lr"],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    return scheduler


def round_probs(probs: torch.Tensor, threshold: float) -> List[int]:
    """Rounds probabilities according to chosen threshold."""
    return (probs > threshold).long()


def compute_top_k(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Computes epoch topK accuracy."""
    _, top_k_indices = torch.topk(probs, list(labels).count(1))
    top_k_labels = torch.Tensor([labels[idx] for idx in top_k_indices])
    top_k_accuracy = torch.mean(top_k_labels == 1, dtype=float)
    return top_k_accuracy.float()


def compute_f1_score(
    epoch_probs: torch.Tensor, epoch_labels: torch.Tensor, prob_threshold: float
) -> float:
    """Computes epoch F1-score."""
    return binary_f1_score(input=epoch_probs, target=epoch_labels, threshold=prob_threshold).float()


def compute_roc_score(epoch_probs: torch.Tensor, epoch_labels: torch.Tensor) -> float:
    """Computes epoch ROC score."""
    return binary_auroc(epoch_probs, epoch_labels).float()


def set_torch_reproducibility(seed: int) -> None:
    """Sets torch and cuda in reproducible mode."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def empty_cache() -> None:
    """Delete model class from memory."""
    torch.cuda.empty_cache()
    gc.collect()
