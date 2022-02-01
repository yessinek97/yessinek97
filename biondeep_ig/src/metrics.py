"""Module used to define the metrics funations."""
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from biondeep_ig.src.logger import get_logger

log = get_logger("Metrics")


def topk_global(labels, scores):
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


def topk_x_observations(data, target_name, prediction_name, observations_number):
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


def global_evaluation(data, target_name, prediction_name):
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
) -> Dict[str, Dict[str, float]]:
    """Per split evaluation."""
    metrics: Dict[str, Any] = defaultdict(dict)
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

        top_k_retrieval_global += metrics[split]["top_k_retrieval"]
        _topk_x, _top_k_retrieval_x = topk_x_observations(
            split_id_df, target_name, prediction_name, observations_number
        )
        (
            metrics[split][f"topk_{observations_number}"],
            metrics[split][f"topk_retrieval_{observations_number}"],
        ) = (
            float(_topk_x),
            int(_top_k_retrieval_x),
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


def train_evaluation(data, target_name, prediction_name):
    """Train evaluation."""
    data[prediction_name].fillna(data[prediction_name].mean(), inplace=True)
    return global_evaluation(data, target_name, prediction_name)


def test_evaluation(data, target_name, prediction_name, eval_id_name, observations_number):
    """Test evaluation."""
    metrics = {}
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


def roc(labels, scores, **args):
    """Compute the AUC score for a given label and predications."""
    return roc_auc_score(labels, scores)


def logloss(labels, scores, **args):
    """Compute the log_loss score for a given label and predications."""
    return log_loss(labels, scores)


def precision(labels, scores, **args):
    """Compute the precision score for a given label and predications."""
    scores = (scores >= args["threshold"]).astype(int)
    return precision_score(labels, scores)


def recall(labels, scores, **args):
    """Compute the recall score for a given label and predications."""
    scores = (scores >= args["threshold"]).astype(int)
    return recall_score(
        labels,
        scores,
    )


def f1(labels, scores, **args):
    """Compute the recall score for a given label and predications."""
    scores = (scores >= args["threshold"]).astype(int)
    return f1_score(labels, scores)


def topk(labels, scores, **args):
    """Compute the topk score for a given label and predications."""
    return topk_global(labels, scores)[0]
