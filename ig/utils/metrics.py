"""Module used to define the metrics functions."""
from logging import Logger
from typing import Any, Dict, Tuple, Union

import pandas as pd
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ig.utils.logger import get_logger

log: Logger = get_logger("Metrics")
EvalDictType = Dict[str, Union[float, int]]
PerSplitEvalDictType = Dict[str, EvalDictType]
TestEvalDictType = Dict[str, Union[EvalDictType, PerSplitEvalDictType]]


def topk_global(labels: Any, scores: Any) -> Tuple[float, int]:
    """Topk global metric."""
    top_k = int(sum(labels))
    assert len(scores) == len(labels)
    if top_k == 0:
        log.info("NO POSITIVE LABEL IN LABELS, quitting")
        return 0.0, 0
    scores_labels = pd.DataFrame({"scores": scores, "labels": labels})
    scores_labels.sort_values(["scores"], inplace=True, kind="mergesort")
    _, labels_out = scores_labels.scores.values, scores_labels.labels.values
    top_k_retrieval = sum(labels_out[-top_k:])
    return float(top_k_retrieval / top_k), int(top_k_retrieval)


def roc(labels: Any, scores: Any, **kwargs: Dict[str, Any]) -> float:  # pylint: disable=W0613
    """Compute the AUC score for a given label and predictions.

    kwargs is used to make the function callable, even if there is an extra argument
    since we call all metric functions in the same way.

    """
    return roc_auc_score(labels, scores)


def logloss(labels: Any, scores: Any, **kwargs: Dict[str, Any]) -> float:  # pylint: disable=W0613
    """Compute the log_loss score for a given label and predictions.

    kwargs is used to make the function callable, even if there is an extra argument
    since we call all metric functions in the same way.

    """
    return log_loss(labels, scores)


def precision(labels: Any, scores: Any, **kwargs: Dict[str, Any]) -> float:  # pylint: disable=W0613
    """Compute the precision score for a given label and predictions."""
    scores = (scores >= kwargs["threshold"]).astype(int)
    return precision_score(labels, scores)


def recall(labels: Any, scores: Any, **kwargs: Dict[str, Any]) -> float:
    """Compute the recall score for a given label and predictions."""
    scores = (scores >= kwargs["threshold"]).astype(int)
    return recall_score(
        labels,
        scores,
    )


def f1(labels: Any, scores: Any, **kwargs: Dict[str, Any]) -> float:
    """Compute the recall score for a given label and predictions."""
    scores = (scores >= kwargs["threshold"]).astype(int)
    return f1_score(labels, scores)


def topk(labels: Any, scores: Any, **kwargs: Dict[str, Any]) -> float:  # pylint: disable=W0613
    """Compute the topk score for a given label and predictions.

    kwargs is used to make the function callable, even if there is an extra argument
    since we call all metric functions in the same way.
    """
    return topk_global(labels, scores)[0]


def roc_auc_curve(labels: Any, scores: Any) -> Tuple[Any, Any]:
    """Compute the Receiver Operating Characteristic (ROC) curve.

    Args:
        labels: The true binary labels.
        scores: The predicted scores.

    Returns:
        Tuple of arrays (fpr, tpr) where fpr is the false positive rates, tpr is the
        true positive rates.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr
