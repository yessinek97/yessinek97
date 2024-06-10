"""Holds functions for evaluation."""
import random
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import log_loss, precision_recall_curve, roc_auc_score, roc_curve

import ig.utils.metrics as src_metrics
from ig import KFOLD_EXP_NAMES, KFOLD_MODEL_NAME, SINGLE_MODEL_NAME
from ig.constants import MetricsEvalType
from ig.utils.cross_validation import load_experiments
from ig.utils.io import load_yml, save_yml
from ig.utils.logger import get_logger
from ig.utils.metrics import EvalDictType, PerSplitEvalDictType, TestEvalDictType, topk, topk_global

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
        split_name: str,
        dataset_name: str,
    ) -> None:
        """Compute metrics for a given data and prediction column.

        Args:
            data: Dataframe object
            prediction_name: str  prediction column name
            split_name: str split name (train,validation,test)
            the comparison score only or all the plots.
            is_plot_fig : #Weather plot Auc Roc curve plot (Validation/test)
            during training or not(True/False)
            dataset_name: The name of the dataset.
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
            if self.eval_id_name and split_name == "test":
                evals_per_id_name = per_split_evaluation(
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
                self.evals[split_name][prediction_name][self.eval_id_name] = evals_per_id_name
            self.evals[split_name][prediction_name]["global"] = prediction_metrics_evaluation
            if self.print_evals:
                log.info("             *%s: %s", split_name, prediction_eval_message)
            if (self.curve_plot_directory is not None) and (self.is_plot_fig):
                self.plot_curve(
                    data=data,
                    prediction_name=prediction_name,
                    plot_path=self.curve_plot_directory
                    / f"{prediction_name}_{dataset_name}_{split_name}_precision_recall_curve.png",
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


def generate_topk_for_exp(folder_path: Path, prediction_name: Optional[str] = None) -> float:
    """Generate the top k for a single of kfold experiment.

    This method calculates the mean prediction for a single
    or kfold experiment across different features
    and different base models ie (xgboost or LGBM)

    Args:
        folder_path: path to experiment
        and the type of experiment used single of kfold.
        prediction_name: prediction in case of  using a single model
        and prediction_mean in case of using a kfold model.

    Returns:
        Mean topk of the ensembling method.

    """
    mean_preds: List[np.array] = []
    for exp_feature in folder_path.iterdir():
        for model_type in (folder_path / exp_feature).iterdir():
            file = pd.read_csv(folder_path / exp_feature / model_type / "prediction/test.csv")
            file = file.dropna(subset=["cd8_any"])
            labels = file["cd8_any"]
            configuration = load_yml(folder_path / exp_feature / model_type / "configuration.yml")
            if not prediction_name:
                prediction_name = configuration["evaluation"]["prediction_name_selector"]
            mean_preds.append(file[prediction_name])
    labels = file["cd8_any"]
    mean_preds_final = np.mean(mean_preds, axis=0)

    return topk(labels, mean_preds_final)


def global_evaluation(data: pd.DataFrame, target_name: str, prediction_name: str) -> EvalDictType:
    """Global evaluation."""
    metrics = {}
    metrics["logloss"] = float(log_loss(data[target_name], data[prediction_name]))
    metrics["auc"] = float(roc_auc_score(data[target_name], data[prediction_name]))
    metrics["topk"], metrics["top_k_retrieval"] = topk_global(
        data[target_name], data[prediction_name]
    )
    return metrics


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


def eval_test_data(
    test_data_path: str,
    single_train_directory: Path,
    single_train_name: str,
    label_name: str,
    predications_eval: Dict[str, List[pd.Series]],
) -> None:
    """Compute metrics for a given file after loading it's prediction."""
    file_name = Path(test_data_path).stem
    file_prediction_path = single_train_directory / "Inference" / f"{file_name}.csv"
    configuration = load_yml(single_train_directory / "best_experiment" / "configuration.yml")
    predictions = pd.read_csv(file_prediction_path)

    if label_name in predictions.columns:
        single_train_results = pd.Series([], dtype=pd.StringDtype())
        prediction_name = configuration["evaluation"]["prediction_name_selector"]
        metrics = global_evaluation(predictions, label_name, prediction_name)
        single_train_results["Single train Name"] = single_train_name
        for key in metrics:
            single_train_results[key] = metrics[key]
        predications_eval[file_name].append(single_train_results)


def experiments_evaluation(
    experiment_paths: List[Path], experiments_path: Path, params: Dict[str, Any]
) -> None:
    """Evaluate each experiment and save the results."""
    experiments_metrics: DefaultDict[str, List[pd.Series]] = defaultdict(list)
    for experiment_path in experiment_paths:
        best_experiment_results_path = experiment_path / "best_experiment" / "eval" / "results.csv"

        if best_experiment_results_path.exists():
            best_experiment_results = pd.read_csv(best_experiment_results_path)
            prediction_name = load_yml(experiment_path / "best_experiment" / "configuration.yml")[
                "evaluation"
            ]["prediction_name_selector"]
            metrics_names = [
                col for col in best_experiment_results.columns if col not in ["split", "prediction"]
            ]
            for metric_name in metrics_names:
                metric_results = pd.Series([], dtype=pd.StringDtype())
                metric_results["metric_name"] = metric_name
                metric_results["experiment_name"] = experiment_path.name
                metric_results["train"] = best_experiment_results[
                    (best_experiment_results.split == "train")
                ][metric_name].mean()
                metric_results["validation"] = best_experiment_results[
                    (best_experiment_results.split == "validation")
                    & (best_experiment_results.prediction == prediction_name)
                ][metric_name].iloc[0]
                metric_results["test"] = best_experiment_results[
                    (best_experiment_results.split == "test")
                    & (best_experiment_results.prediction == prediction_name)
                ][metric_name].iloc[0]
                experiments_metrics[metric_name].append(metric_results)

    experiments_metrics_dfs: Dict[str, pd.DataFrame] = {}
    for metric_name in metrics_names:
        experiments_metrics_dfs[metric_name] = pd.DataFrame(experiments_metrics[metric_name])
        experiments_metrics_dfs[metric_name].to_csv(
            experiments_path / "results" / f"{metric_name}.csv", index=False
        )
    display_metrics = [
        metric_name
        for metric_name in params["metrics"]
        if metric_name in experiments_metrics_dfs.keys()
    ]
    for metric_name in display_metrics:
        log.info("%s evaluation: ############################################", metric_name)
        for line in experiments_metrics_dfs[metric_name].to_string().split("\n"):
            log.info("%s", line)


def parse_comparison_score_metrics_to_df(
    metrics: Dict[str, MetricsEvalType], comparison_score: str
) -> pd.DataFrame:
    """Parse comparison score metrics to dataframe object."""
    results = []
    for key, values in metrics.items():
        for prediction, scores in values.items():
            result = {}
            result["experiments"] = key
            splites_pred = prediction.split("_")
            result["prediction"] = (
                "prediction" if len(splites_pred) == 1 else f"prediction_{splites_pred[-1]}"
            )
            result["split"] = prediction if len(splites_pred) == 0 else splites_pred[0]
            result["topk"] = scores[comparison_score]["global"]["topk"]
            result["roc"] = scores[comparison_score]["global"]["roc"]
            results.append(result)

    results_cs = pd.DataFrame(results)
    results_cs["type"] = comparison_score
    return results_cs


def eval_comparison_score(
    configuration: Dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    dataset_name: str,
    experiment_path: Path,
    plot_comparison_score_only: bool = True,
) -> Optional[pd.DataFrame]:  # noqa
    """Eval comparison score."""
    comparison_score = configuration["evaluation"].get("comparison_score", None)
    if comparison_score:
        log.info("Eval %s", comparison_score)
        results: Dict[str, MetricsEvalType] = {}

        experiment_names, _ = load_experiments(configuration)
        if SINGLE_MODEL_NAME in experiment_names:
            validation_column = configuration["experiments"][SINGLE_MODEL_NAME]["validation_column"]
            if validation_column in train_data.columns:

                curve_plot_directory = experiment_path / "comparison_score" / SINGLE_MODEL_NAME
                curve_plot_directory.mkdir(exist_ok=True, parents=True)
                evaluator = Evaluation(
                    label_name=configuration["label"],
                    eval_configuration=configuration["evaluation"],
                    curve_plot_directory=curve_plot_directory,
                    plot_comparison_score_only=plot_comparison_score_only,
                )

                log.info(SINGLE_MODEL_NAME)

                evaluator.compute_metrics(
                    data=train_data[train_data[validation_column] == 0],
                    prediction_name=comparison_score,
                    split_name="train",
                    dataset_name=dataset_name,
                )
                evaluator.compute_metrics(
                    data=train_data[train_data[validation_column] == 1],
                    prediction_name=comparison_score,
                    split_name="validation",
                    dataset_name=dataset_name,
                )
                results[SINGLE_MODEL_NAME] = evaluator.get_evals()
        kfold_exps = list(set(experiment_names) & set(KFOLD_EXP_NAMES))
        if kfold_exps:
            split_column = configuration["experiments"][kfold_exps[0]]["split_column"]
            if split_column in train_data.columns:

                log.info(KFOLD_MODEL_NAME)
                curve_plot_directory = experiment_path / "comparison_score" / KFOLD_MODEL_NAME
                curve_plot_directory.mkdir(exist_ok=True, parents=True)
                evaluator = Evaluation(
                    label_name=configuration["label"],
                    eval_configuration=configuration["evaluation"],
                    curve_plot_directory=curve_plot_directory,
                    plot_comparison_score_only=plot_comparison_score_only,
                )
                for split in np.sort(train_data[split_column].unique()):
                    log.info(split)
                    evaluator.compute_metrics(
                        data=train_data[train_data[split_column] != split],
                        prediction_name=comparison_score,
                        split_name=f"train_{split}",
                        dataset_name=dataset_name,
                    )
                    evaluator.compute_metrics(
                        data=train_data[train_data[split_column] == split],
                        prediction_name=comparison_score,
                        split_name=f"validation_{split}",
                        dataset_name=dataset_name,
                    )
                results[KFOLD_MODEL_NAME] = evaluator.get_evals()
        log.info("Test")
        curve_plot_directory = experiment_path / "comparison_score"
        evaluator = Evaluation(
            label_name=configuration["label"],
            eval_configuration=configuration["evaluation"],
            curve_plot_directory=curve_plot_directory,
            plot_comparison_score_only=plot_comparison_score_only,
        )
        evaluator.compute_metrics(
            data=test_data,
            prediction_name=comparison_score,
            split_name="test",
            dataset_name=dataset_name,
        )
        results["test"] = evaluator.get_evals()
        save_yml(
            results,
            experiment_path / "comparison_score" / f"eval_{comparison_score}.yml",
        )

        return parse_comparison_score_metrics_to_df(results, comparison_score)
    return None
