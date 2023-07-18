"""Holds functions for evaluation."""
import random
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import precision_recall_curve, roc_curve

import ig.src.metrics as src_metrics
from ig import KFOLD_EXP_NAMES
from ig.constants import MetricsEvalType
from ig.src.logger import get_logger

log: Logger = get_logger("Evaluation")


class Evaluation:
    """Class to handle evaluation.

    Attributes:
        label_name: target name
        eval_configuration: the fll eval configuration
        curve_plot_directory: path where  the curve plot will be saved
        is_compute_metrics: whether to compute the prediction metrics.
    """

    def __init__(
        self,
        label_name: str,
        eval_configuration: Dict[str, Any],
        curve_plot_directory: Optional[Path],
        is_compute_metrics: bool = True,
        plot_comparison_score_only: bool = False,
    ):
        """Initialize the Evaluation class."""
        self.label_name = label_name
        self.eval_configuration = eval_configuration
        self.curve_plot_directory = curve_plot_directory
        self.metrics = self.get_metrics_from_string()
        self.evals: MetricsEvalType = defaultdict(lambda: defaultdict(dict))
        self.is_compute_metrics = is_compute_metrics
        self.prediction_name_selector = self.eval_configuration.get(
            "prediction_name_selector", None
        )
        self.print_evals = self.eval_configuration.get("print_evals", False)
        self.plot_comparison_score_only = plot_comparison_score_only
        self.is_plot_fig = True

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
    def comparison_score(self) -> str:
        """Get the comparison_score name from configuration."""
        return self.eval_configuration.get("comparison_score", None)

    def set_is_plot_fig(self, flag: bool) -> None:
        """Reset is plot fig option."""
        self.is_plot_fig = flag

    def reset_print_evals(self) -> None:
        """Reset the print evals varible to the default one."""
        self.print_evals = self.eval_configuration.get("print_evals", False)

    def get_metrics_from_string(self) -> Dict[str, Callable]:
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
                        "%s is not implemented in sklearn"
                        "or in the defined metrics.It will not be used",
                        metric_name,
                    )
        if len(metrics) == 0:
            raise NotImplementedError(
                "No metric is defined,please chose one"
                "from the available list "
                ":roc ,logloss ,precession, recall, topk,f1."
            )
        return metrics

    def compute_metrics(
        self,
        data: pd.DataFrame,
        prediction_name: str,
        data_name: str,
    ) -> None:
        """Compute metrics for a given data and prediction column.

        Args:
            data: Dataframe object
            prediction_name: str  prediction column name
            data_name: str split name (train,validation,test)
            the comparison score only or all the plots.
            is_plot_fig : #Weather plot Auc Roc curve plot (Validation/test)
            during training or not(True/False)
        """
        if self.comparison_score:
            # Impute the missing values for comparison score column before plotting metrics
            data[self.comparison_score] = data[self.comparison_score].fillna(
                data[self.comparison_score].mean()
            )
        if self.is_compute_metrics:

            prediction_metrics_evaluation = {}
            prediction_eval_message = ""
            data = data.loc[(data[self.label_name] == 0) | (data[self.label_name] == 1)]

            for metric_name, metric_fuc in self.metrics.items():
                prediction_metrics_evaluation[metric_name] = float(
                    metric_fuc(
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
                log.info("             *%s: %s", data_name, prediction_eval_message)
            if (self.curve_plot_directory is not None) and (self.is_plot_fig):
                self.plot_curve(
                    data=data,
                    prediction_name=prediction_name,
                    plot_path=self.curve_plot_directory
                    / f"{prediction_name}_{data_name}_precision_recall_curve.png",
                    prediction_metrics_evaluation=prediction_metrics_evaluation,
                )

    def get_evals(self) -> MetricsEvalType:
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
    ) -> pd.DataFrame:
        """Return statistic kfold scores."""
        test_metrics = self.evals["test"]
        statistic_scores = pd.Series([], dtype=pd.StringDtype())
        for opt in statistic_opt:
            statistic_scores[f"{opt}_{self.metric_selector}"] = getattr(np, opt)(
                [test_metrics[e]["global"][self.metric_selector] for e in prediction_columns_name]
            )
        statistic_scores["ID"] = f"{experiment_name}//{features_name}//{model_type}"
        return pd.DataFrame([statistic_scores])

    def get_experiment_best_scores(
        self,
        results: pd.DataFrame,
        experiment_name: str,
        model_type: str,
        features_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """Return best experiment scores."""
        metrics = results[results["split"] == self.data_name_selector]
        metrics.sort_values(
            [self.metric_selector], inplace=True, ascending=self.metrics_selector_higher
        )
        if (experiment_name in KFOLD_EXP_NAMES) and (self.prediction_name_selector):
            best_prediction_name = self.prediction_name_selector
        else:
            best_prediction_name = metrics.iloc[-1]["prediction"]
            self.prediction_name_selector = best_prediction_name
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

        return best_validation_scores, best_test_scores, best_prediction_name

    def plot_curve(
        self,
        data: pd.DataFrame,
        prediction_name: str,
        plot_path: Path,
        prediction_metrics_evaluation: Dict[str, float],
    ) -> None:
        """Plot precision recall curve and roc curve."""
        _, ax = plt.subplots(figsize=(14, 8), ncols=2)

        if self.comparison_score:

            # Plot the precision-recall curves for the Comparison Score column only

            cs_precision, cs_recall, _ = precision_recall_curve(
                data[self.label_name], data[self.comparison_score]
            )
            cs_fpr, cs_tpr, _ = roc_curve(data[self.label_name], data[self.comparison_score])
            ax[0].plot(cs_recall, cs_precision, label=f"{self.comparison_score}_precision_recall")
            ax[1].plot(cs_fpr, cs_tpr, label=f"{self.comparison_score}_roc")

        if not self.plot_comparison_score_only:
            precision, recall, _ = precision_recall_curve(
                data[self.label_name], data[prediction_name]
            )
            fpr, tpr, _ = roc_curve(data[self.label_name], data[prediction_name])
            # Create random classifier probabilities of the positive outcome for the roc curve
            random_classifier_probabilities_roc = [
                random.random() for _ in range(len(data[self.label_name]))
            ]
            # Create random classifier equation of the positive outcome for precision-recall curve

            random_fpr, random_tpr, _ = roc_curve(
                data[self.label_name], random_classifier_probabilities_roc
            )
            random_classifier_pr_curve = len(
                data[self.label_name][data[self.label_name] == 1]
            ) / len(data[self.label_name])
            ax[0].plot(recall, precision, label="IG_model_precision_recall")
            ax[1].plot(fpr, tpr, label="IG_model_roc")
            ax[0].plot(
                [0, 1],
                [random_classifier_pr_curve, random_classifier_pr_curve],
                linestyle="--",
                label="Random_model_precision_recall",
            )
            ax[1].plot(random_fpr, random_tpr, linestyle="--", label="Random_model_roc")

        ax[0].set_xlabel("Recall")
        ax[0].set_ylabel("Precision")
        ax[0].legend(loc="upper right")
        ax[0].set_title("precision-recall plot")
        ax[1].set_xlabel("False positive rate")
        ax[1].set_ylabel("True positive rate")
        ax[1].legend(loc="upper left")
        ax[1].set_title("AUC ROC curve plot")
        plt.suptitle(
            prediction_name
            + ": TopK Score: "
            + str("{:.3f}".format(prediction_metrics_evaluation["topk"]))
            + " ROC Score: "
            + str("{:.3f}".format(prediction_metrics_evaluation["roc"])),
            fontsize=16,
        )
        plt.savefig(plot_path)
        plt.clf()
        plt.close()
