"""Holds functions for evaluation."""
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

import biondeep_ig.src.metrics as src_metrics
from biondeep_ig.src import DKFOLD_MODEL_NAME
from biondeep_ig.src import Evals
from biondeep_ig.src import KFOLD_MODEL_NAME
from biondeep_ig.src.logger import get_logger

log = get_logger("Evaluation")


class Evaluation:
    """Class to handle evaluation.

    Attributes:
        label_name: target name
        eval_configuration: the fll eval configuration
        curve_plot_directory: path where  the curve plot will be saved
    """

    def __init__(
        self, label_name: str, eval_configuration: Dict[str, Any], curve_plot_directory: Path
    ):
        """Initialize the Evaluation class."""
        self.label_name = label_name
        self.eval_configuration = eval_configuration
        self.curve_plot_directory = curve_plot_directory
        self.metrics = self.get_metrics_from_string()
        self.evals: Evals = defaultdict(lambda: defaultdict(dict))

    @property
    def eval_id_name(self) -> str:
        """Get the eval_id_name variable."""
        return self.eval_configuration.get("eval_id_name", None)

    @property
    def observations_number(self) -> int:
        """Get the observations_number variable."""
        return self.eval_configuration.get("observations_number", 20)

    @property
    def str_metrics(self) -> List[str]:
        """Get the str_metrics variable."""
        return self.eval_configuration.get("metrics", ["auc", "topk"])

    @property
    def threshold(self) -> float:
        """Get the threshold variable."""
        return self.eval_configuration.get("threshold", 0.5)

    @property
    def print_evals(self) -> bool:
        """Get the print_evals variable."""
        return self.eval_configuration.get("print_evals", False)

    @property
    def metric_selector(self) -> str:
        """Get the metric_selector variable."""
        return self.eval_configuration.get("metric_selector", "topk")

    @property
    def metrics_selector_higher(self) -> bool:
        """Get the observations_number variable."""
        return self.eval_configuration.get("metrics_selector_higher", True)

    @property
    def data_name_selector(self) -> str:
        """Get the data_name_selector variable."""
        return self.eval_configuration.get("data_name_selector", "test")

    @property
    def prediction_name_selector(self) -> str:
        """Get the prediction_name_selector variable."""
        return self.eval_configuration.get("prediction_name_selector", None)

    def get_metrics_from_string(self) -> Dict[str, Any]:
        """Import metrics from string name."""
        metrics = {}
        for metric_name in self.str_metrics:
            try:
                metrics[metric_name] = getattr(src_metrics, metric_name)
            except AttributeError:
                try:
                    metrics[metric_name] = getattr(sklearn_metrics, metric_name)
                except AttributeError:
                    log.warning(
                        f"{metric_name} is not implemented in sklearn or in the defined metrics.It will not be used"
                    )
        if len(metrics) == 0:
            raise NotImplementedError(
                "No metric is defined,please chose one from the available list :roc ,logloss ,precession, recall, topk,f1."
            )
        return metrics

    def compute_metrics(self, data: pd.DataFrame, prediction_name: str, data_name: str) -> None:
        """Compute metrics for a given data and prediction column.

        Args:
            data: Dataframe object
            prediction_name: str  prediction column name
            data_name: str split name (train,validation,test)
        """
        prediction_metrics_evaluation = {}
        prediction_eval_message = ""
        data = data.loc[(data[self.label_name] == 0) | (data[self.label_name] == 1)]
        for metric_name in self.metrics:
            prediction_metrics_evaluation[metric_name] = float(
                self.metrics[metric_name](
                    labels=data[self.label_name],
                    scores=data[prediction_name],
                    threshold=self.threshold,
                )
            )
            prediction_eval_message += (
                f"{metric_name}: {prediction_metrics_evaluation[metric_name]:0.3f} "
            )
        if self.eval_id_name and data_name == "test":
            evals_per_id_name = src_metrics.per_split_evaluation(
                data=data,
                target_name=self.label_name,
                prediction_name=prediction_name,
                eval_id_name=self.eval_id_name,
                metrics_list=self.metrics,
                threshold=self.threshold,
                observations_number=self.observations_number,
            )

            global_per_id_name = evals_per_id_name["global"]

            metric_name = f"topk_{self.observations_number}_{self.eval_id_name}"
            prediction_eval_message += f"{metric_name}: {global_per_id_name[metric_name]:0.3f} "
            metric_name = f"topk_{self.eval_id_name}"
            prediction_eval_message += f"{metric_name}: {global_per_id_name[metric_name]:0.3f} "

            prediction_metrics_evaluation.update(global_per_id_name)
            self.evals[data_name][prediction_name][self.eval_id_name] = evals_per_id_name

        self.evals[data_name][prediction_name]["global"] = prediction_metrics_evaluation
        if self.print_evals:
            log.info(f"             *{data_name}: {prediction_eval_message}")
        if self.curve_plot_directory:
            self.plot_curve(
                data=data,
                prediction_name=prediction_name,
                plot_path=self.curve_plot_directory
                / f"{prediction_name}_{data_name}_precision_recall_curve.png",
            )

    def get_evals(self) -> Evals:
        """Return evaluation dict."""
        for key in self.evals:
            self.evals[key] = dict(self.evals[key])
        return dict(self.evals)

    def get_statistic_kfold_scores(
        self,
        statistic_opt: List[str],
        prediction_columns_name: List[str],
        experiment_name: str,
        model_type: str,
        features_name: str,
    ) -> Dict[str, float]:
        """Return statistic kfold scores."""
        test_metrics = self.evals["test"]
        statistic_scores = {}
        for opt in statistic_opt:
            statistic_scores[f"{opt}_{self.metric_selector}"] = getattr(np, opt)(
                [test_metrics[e]["global"][self.metric_selector] for e in prediction_columns_name]
            )
        statistic_scores["ID"] = f"{experiment_name}//{features_name}//{model_type}"
        return statistic_scores

    def get_experiment_best_scores(
        self, results: pd.DataFrame, experiment_name: str, model_type: str, features_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return best experiment scores."""
        metrics = results[results["split"] == self.data_name_selector]
        metrics.sort_values(
            [self.metric_selector], inplace=True, ascending=self.metrics_selector_higher
        )
        if (experiment_name in [KFOLD_MODEL_NAME, DKFOLD_MODEL_NAME]) and (
            self.prediction_name_selector
        ):
            best_prediction_name = self.prediction_name_selector
        else:
            best_prediction_name = metrics.iloc[-1]["prediction"]
        best_validation_scores = results[
            (results["prediction"] == best_prediction_name) & (results["split"] == "validation")
        ]
        best_test_scores = results[
            (results["prediction"] == best_prediction_name) & (results["split"] == "test")
        ]

        best_validation_scores["experiment"] = experiment_name
        best_validation_scores["model"] = model_type
        best_validation_scores["features"] = features_name
        best_validation_scores["ID"] = f"{experiment_name}//{features_name}//{model_type}"

        best_test_scores["experiment"] = experiment_name
        best_test_scores["model"] = model_type
        best_test_scores["features"] = features_name
        best_test_scores["ID"] = f"{experiment_name}//{features_name}//{model_type}"

        return best_validation_scores, best_test_scores

    def plot_curve(self, data: pd.DataFrame, prediction_name: str, plot_path: Path) -> None:
        """Plot precision recall curve and roc curve."""
        precision, recall, thresholds = precision_recall_curve(
            data[self.label_name], data[prediction_name]
        )
        fpr, tpr, thresholds = roc_curve(data[self.label_name], data[prediction_name])
        fig, ax = plt.subplots(figsize=(12, 6), ncols=2)
        ax[0].plot(recall, precision, label="precision_recall_curve")
        ax[1].plot(fpr, tpr, label="roc_curve")

        ax[0].set_xlabel("Recall")
        ax[0].set_ylabel("Precision")
        ax[0].legend(loc="upper right")
        ax[1].set_xlabel("False positive rate")
        ax[1].set_ylabel("True positive rate")
        ax[1].legend(loc="upper left")
        fig.suptitle(prediction_name, fontsize=16)
        plt.savefig(plot_path)
        plt.clf()
        plt.close()
