"""Module used to define the metrics funations."""
from collections import defaultdict
from logging import Logger
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score

from ig.src.logger import get_logger

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


def topk_x_observations(
    data: pd.DataFrame, target_name: str, prediction_name: str, observations_number: int
) -> Tuple[float, int]:
    """Return the positive labels captured per subdata."""
    top_k = int(data[target_name].sum())
    top_k = np.minimum(top_k, observations_number)
    if top_k == 0:
        log.info("NO POSITIVE LABEL IN LABELS, quitting")
        return 0.0, 0

    top_k_retrieval = data.sort_values(prediction_name, ascending=False)[:observations_number][
        target_name
    ].sum()
    return float(top_k_retrieval / top_k), top_k_retrieval


def global_evaluation(data: pd.DataFrame, target_name: str, prediction_name: str) -> EvalDictType:
    """Global evaluation."""
    metrics = {}
    metrics["logloss"] = float(log_loss(data[target_name], data[prediction_name]))
    metrics["auc"] = float(roc_auc_score(data[target_name], data[prediction_name]))
    metrics["topk"], metrics["top_k_retrieval"] = topk_global(
        data[target_name], data[prediction_name]
    )
    return metrics


def per_split_evaluation(
    data: pd.DataFrame,
    target_name: str,
    prediction_name: str,
    eval_id_name: str,
    observations_number: int = 20,
    metrics_list: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
) -> PerSplitEvalDictType:
    """Per split evaluation."""
    metrics: PerSplitEvalDictType = defaultdict(dict)
    top_k_retrieval_x = 0
    top_k_retrieval_global = 0
    effective_true_label_x = 0
    for _, split_id_df in data.groupby(eval_id_name):
        split = split_id_df[eval_id_name].unique()[0]
        if metrics_list:
            for metric_name in metrics_list:
                metrics[split][metric_name] = float(
                    metrics_list[metric_name](
                        labels=split_id_df[target_name],
                        scores=split_id_df[prediction_name],
                        threshold=threshold,
                    )
                )
            metrics[split]["topk"], metrics[split]["top_k_retrieval"] = topk_global(
                split_id_df[target_name], split_id_df[prediction_name]
            )
        else:
            metrics[split] = global_evaluation(
                data=split_id_df, target_name=target_name, prediction_name=prediction_name
            )

        top_k_retrieval_global += int(metrics[split]["top_k_retrieval"])
        split_topk_x, split_top_k_retrieval_x = topk_x_observations(
            split_id_df, target_name, prediction_name, observations_number
        )
        (
            metrics[split][f"topk_{observations_number}"],
            metrics[split][f"topk_retrieval_{observations_number}"],
        ) = (
            float(split_topk_x),
            int(split_top_k_retrieval_x),
        )
        metrics[split]["true_label"] = int(split_id_df[target_name].sum())
        top_k_retrieval_x += int(metrics[split][f"topk_retrieval_{observations_number}"])
        effective_true_label_x += np.minimum(metrics[split]["true_label"], observations_number)
    metrics["global"] = {}
    metrics["global"][f"topk_retrieval_{observations_number}_{eval_id_name}"] = int(
        top_k_retrieval_x
    )
    metrics["global"][f"topk_retrieval_{eval_id_name}"] = int(top_k_retrieval_global)

    metrics["global"][f"topk_{observations_number}_{eval_id_name}"] = float(
        top_k_retrieval_x / effective_true_label_x
    )
    metrics["global"][f"topk_{eval_id_name}"] = float(
        top_k_retrieval_global / data[target_name].sum()
    )

    return dict(metrics)


def train_evaluation(data: pd.DataFrame, target_name: str, prediction_name: str) -> EvalDictType:
    """Train evaluation."""
    data[prediction_name].fillna(data[prediction_name].mean(), inplace=True)
    return global_evaluation(data, target_name, prediction_name)


def test_evaluation(
    data: pd.DataFrame,
    target_name: str,
    prediction_name: str,
    eval_id_name: str,
    observations_number: int,
) -> TestEvalDictType:
    """Test evaluation."""
    metrics: TestEvalDictType = {}
    metrics["global"] = global_evaluation(data, target_name, prediction_name)
    if target_name == "cd8_any":
        metrics[f"per_{eval_id_name}"] = per_split_evaluation(
            data=data,
            target_name=target_name,
            prediction_name=prediction_name,
            eval_id_name=eval_id_name,
            observations_number=observations_number,
        )
    return metrics


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
