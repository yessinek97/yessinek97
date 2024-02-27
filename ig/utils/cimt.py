"""Module de define a helper functions for cimt command."""
from collections import defaultdict
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import click
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ig.src.experiments as exper
from ig import FEATURES_DIRECTORY
from ig.src.dataset import Dataset
from ig.src.logger import get_logger
from ig.src.metrics import logloss, roc_auc_curve, roc_auc_score, topk
from ig.src.utils import import_experiment, load_pkl, load_yml, read_data, save_yml

log: Logger = get_logger("utils/cimt")
CsEvalType = Dict[str, Dict[str, Any]]


def features_importance_extraction(
    base_exp_path: Path,
    exp_name: str,
    n_splits: int,
    train_fold: int,
    ntop_features: int,
    do_w_train: bool,
) -> None:
    """Extracts features importance from different splits and saves the N top features.

    Args:
        base_exp_path (Path): The base path where the experiment is located.
        exp_name (str): The name of the experiment.
        n_splits (int): The number of splits to extract features importance from.
        train_fold (int): The train fold number.
        ntop_features (int): The number of top features to save.
        do_w_train (bool): Whether to save the features for training set.

    Returns:
        None
    """
    features_importance_per_split = []
    for i in range(n_splits):
        split_exp_path = base_exp_path / f"split_{i}"
        features_importance_per_split.append(
            get_split_features_importance(split_exp_path, train_fold)
        )

    features_importance = pd.concat(features_importance_per_split)
    features_importance_mean = (
        features_importance.groupby("features")
        .value.mean()
        .rename("value")
        .reset_index()
        .sort_values("value")
    )
    features_importance_count = (
        features_importance.groupby("features").features.count().rename("count").reset_index()
    )
    features_importance = features_importance_mean.merge(
        features_importance_count, on="features", how="left"
    )
    features_importance["count"] /= n_splits
    features_importance["weight"] = features_importance.value * features_importance["count"]
    features_importance.sort_values("weight", inplace=True)
    list_of_important_features = features_importance[-ntop_features:].features.tolist()
    if do_w_train:
        features_path = FEATURES_DIRECTORY / "CD8" / f"{exp_name}_features.txt"
        log.info(f"Save the Top {ntop_features} features to {features_path} ")
        save_features(list_of_important_features, features_path)
    features_path = base_exp_path / "Weighted_features.txt"
    log.info(f"Save the Top {ntop_features} features to {features_path} ")
    save_features(list_of_important_features, features_path)
    features_importance.to_csv(base_exp_path / "features_importance.csv", index=False)


def get_split_features_importance(base_exp_path: Path, train_fold: int) -> pd.DataFrame:
    """Get features importance per split.

    Get the feature importance for each split, concatenate them and return
    the average importance per feature.

    Args:
        base_exp_path (Path): The base experiment path.
        train_fold (int): The number of train folds.

    Returns:
        pd.DataFrame: The features importance per split.
    """
    importance_features_list = []
    for i in range(train_fold):
        path = base_exp_path / "best_experiment" / "checkpoint" / f"split_{i}" / "model.pkl"
        model = load_pkl(path)
        features_importance = model.model.get_score(importance_type="gain")
        importance_features_list.append(
            pd.DataFrame.from_dict(
                {"features": features_importance.keys(), "value": features_importance.values()}
            )
        )
        # save per split ##################################
    importance_features = pd.concat(importance_features_list)
    importance_features = (
        importance_features.groupby("features").value.mean().rename("value").reset_index()
    )
    return importance_features


def save_features(features: List[str], file_path: Path) -> None:
    """Save a list of features as a text format.

    Each element of the list is written in a new line in the file.

    Args:
        features (List[str]): The list of features to save.
        file_path (Path): The path of the file where to save the features.
    """
    with open(file_path, "w") as f:
        for e in features:
            f.write(str(e) + "\n")


def get_experiment_average_score(
    base_exp_path: Path, label_name: str, n_splits: int, comparison_score: str
) -> None:
    """Extracts and merges the prediction and evaluation per split and reports average results.

    Args:
        base_exp_path (Path): The base path of the experiment.
        label_name (str): The name of the label column in the dataset.
        n_splits (int): The number of splits in the experiment.
        comparison_score (Optional[str]): The comparison score to evaluate against
        (default is None).
    """
    splits_eval = []
    splits_pred = []
    comparison_score_eval = []
    for split in range(n_splits):
        series_model = pd.Series([], dtype=pd.StringDtype())
        exp_path = base_exp_path / f"split_{split}"

        split_eval_df = pd.read_csv(exp_path / "best_experiment" / "eval" / "results.csv")
        split_prediction_df = pd.read_csv(exp_path / "best_experiment" / "prediction" / "test.csv")
        split_eval_df_pred_mean = split_eval_df[split_eval_df.prediction == "prediction_mean"]
        series_model["name"] = f"split_{split}"
        series_model["Train"] = split_eval_df[split_eval_df.split == "train"].topk.mean()
        series_model["Validation"] = split_eval_df_pred_mean[
            split_eval_df_pred_mean.split == "validation"
        ].topk.iloc[0]
        series_model["Test"] = split_eval_df_pred_mean[
            split_eval_df_pred_mean.split == "test"
        ].topk.iloc[0]
        splits_eval.append(series_model)
        splits_pred.append(split_prediction_df)
        if comparison_score:
            comparison_score_file = exp_path / "comparison_score" / f"eval_{comparison_score}.yml"
            if comparison_score_file.exists():
                comparison_score_eval.append(
                    load_comparison_score_evaluation(
                        comparison_score_file, f"split_{split}", comparison_score
                    )
                )
    eval_df = pd.DataFrame(splits_eval)
    predictions = pd.concat(splits_pred)
    test_global_topk = topk(predictions[label_name], predictions.prediction_mean)
    average_train_topk = eval_df.Train.mean()
    average_validation_topk = eval_df.Validation.mean()
    average_test_topk = eval_df.Test.mean()

    log.info("Average Splits topk")
    log.info(f" -Train : {average_train_topk:.3}")
    log.info(f" -Validation : {average_validation_topk:.3}")
    log.info(f" -Test : {average_test_topk:.3}")
    log.info("Global topk")
    log.info(f" -Test : {test_global_topk:.3}")
    eval_df.to_csv(base_exp_path.parent / "eval_per_split.csv", index=False)
    predictions.to_csv(base_exp_path.parent / "predictions.csv", index=False)
    results = {
        "Average": {
            "Train": float(average_train_topk),
            "Validation": float(average_validation_topk),
            "Test": float(average_test_topk),
        },
        "Global": {"Test": float(test_global_topk)},
    }
    if comparison_score_eval:
        comparison_score_eval_df = pd.DataFrame(comparison_score_eval)
        average_comparison_score_test_topk = comparison_score_eval_df.Test.mean()
        log.info(f"Eval {comparison_score}")
        log.info("Average Splits topk")
        log.info(f" -Test : {average_comparison_score_test_topk:.3}")
        cs_results = {
            "comparison score": {
                "Average": {
                    "Test": float(average_comparison_score_test_topk),
                },
            }
        }
        if comparison_score in predictions.columns:
            global_comparison_score_test_topk = topk(
                predictions[label_name], predictions[comparison_score]
            )
            log.info("Global topk")
            log.info(f" -Test : {global_comparison_score_test_topk:.3}")
            cs_results["comparison score"].update({"Global": {"Test": float(test_global_topk)}})
    save_yml(results, base_exp_path.parent / "scores.yml")


def load_comparison_score_evaluation(
    comparison_score_file: Path, split: str, comparison_score: str
) -> pd.Series:
    """Load and extract comparison score from comparison score evaluation.

    Load the comparison score evaluation from the given yml file and extract the
    global topk for the given comparison_score on the given split.

    Args:
        comparison_score_file: Path to the comparison score yml file.
        split: Split name.
        comparison_score: Comparison score name.

    Returns:
        A pandas Series containing the split and its corresponding global topk
        for the given comparison_score in the test set.
    """
    eval_score = load_yml(comparison_score_file)
    series = pd.Series([], dtype=pd.StringDtype())
    series["split"] = split
    series["Test"] = eval_score["test"]["test"][comparison_score]["global"]["topk"]
    return series


def load_split_experiment_dataloader(
    ctx: Union[click.core.Context, Any],
    split_experiment_path: Path,
    data_path: str,
) -> Tuple[Any, Dataset]:
    """Load Experiment and dataloader for the given split_experiment_path and data_path.

    Args:
        ctx (Union[click.core.Context, Any]): The context for the experiment
        split_experiment_path (Path): The path to the split experiment
        data_path (str): The path to the data

    Returns:
        Tuple[Any, Dataset]: The loaded experiment and the dataset
    """
    best_exp_path = split_experiment_path / "best_experiment"

    exp_configuration = load_yml(best_exp_path / "configuration.yml")
    experiment_name = list(exp_configuration["experiments"])[0]
    experiment_params = exp_configuration["experiments"][experiment_name]
    features_file_path = best_exp_path / "features.txt"
    data_loader = Dataset(
        click_ctx=ctx,
        data_path=data_path,
        configuration=exp_configuration,
        is_train=False,
        experiment_path=best_exp_path.parent,
        force_gcp=True,
        is_inference=True,
        process_label=False,
    ).load_data()
    experiment_class = import_experiment(exper, experiment_name)
    experiment = experiment_class(
        train_data=None,
        test_data=data_loader,
        configuration=exp_configuration,
        experiment_name=experiment_name,
        folder_name=None,
        sub_folder_name=None,
        experiment_directory=best_exp_path,
        features_file_path=features_file_path,
        **experiment_params,
    )
    return experiment, data_loader


def aggregate_predictions(
    data: pd.DataFrame,
    split_predictions: List[pd.DataFrame],
    ids: List[str],
    features: List[str],
    prediction_columns_name: str,
    label_name: str,
) -> pd.DataFrame:
    """Aggregate predictions in order to return average predictions of different splits.

    This function takes a list of predictions dataframes, a list of identifying column names,
    a list of feature column names, the name of the prediction column and the name of the label
    column.It merges the predictions and return an aggregated dataframe containing the average
    predictions for each split on the identifying columns and the features.
    The merged dataframe also contains the label column.

    Args:
        data (pd.DataFrame): The original data to merge the predictions on.
        split_predictions (List[pd.DataFrame]): The list of predictions dataframes to aggregate.
        ids (List[str]): The list of identifying column names to group on.
        features (List[str]): The list of feature column names to keep.
        prediction_columns_name (str): The name of the prediction column.
        label_name (str): The name of the label column.

    Returns:
        pd.DataFrame: The aggregated dataframe.
    """
    label_pred = pd.concat(split_predictions)[ids + [prediction_columns_name]]
    label_pred = label_pred.groupby(ids)[[prediction_columns_name]].mean().reset_index()
    columns = ids + features + [label_name]
    data = data[[col for col in columns if col in data.columns]]
    test_data = data.merge(label_pred, on=ids, how="left")
    return test_data


def update_configuration(
    default_configuration: Dict[str, Any], experiment_configuration: Dict[str, Any]
) -> Dict[str, Any]:
    """Update the default configuration file with the given experiment configuration."""
    configuration = deepcopy(default_configuration)
    for key in experiment_configuration.keys():
        if isinstance(experiment_configuration[key], Dict):
            for second_key in experiment_configuration[key].keys():
                key_configuration = configuration[key]
                key_configuration[second_key] = experiment_configuration[key][second_key]
            configuration[key] = key_configuration
        else:
            configuration[key] = experiment_configuration[key]

    return configuration


def experiments_evaluation(
    sub_experiments_path: List[Path],
    experiment_path: Path,
    eval_params: Dict[str, Any],
    comparison_columns: List[str],
    cs_evaluations: Dict[str, CsEvalType],
    label: str,
) -> None:
    """Evaluate experiments and store the metrics in CSV files.

    Args:
        sub_experiments_path (List[Path]): List of paths to sub-experiment directories.
        experiment_path (Path): Path to the main experiment directory.
        eval_params (Dict[str, Any]): Dictionary of evaluation parameters.

    Returns:
        None
    """
    metrics_names: List[str]
    experiments_roc_curve: Dict[str, Dict[str, Any]] = {}
    experiments_metrics: DefaultDict[str, List[pd.Series]] = defaultdict(list)
    list_tests_df: List[pd.DataFrame] = []
    for sub_experiment_path in sub_experiments_path:
        best_experiment = sub_experiment_path / "best_experiment"
        best_experiment_results_path = best_experiment / "eval" / "results.csv"

        if best_experiment.exists():

            best_experiment_results = pd.read_csv(best_experiment_results_path)
            prediction_name = load_yml(
                sub_experiment_path / "best_experiment" / "configuration.yml"
            )["evaluation"]["prediction_name_selector"]
            metrics_names = [
                col for col in best_experiment_results.columns if col not in ["split", "prediction"]
            ]
            for metric_name in metrics_names:
                metric_results = pd.Series([], dtype=pd.StringDtype())
                metric_results["metric_name"] = metric_name
                metric_results["experiment_name"] = sub_experiment_path.name
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

            best_exp_prediction_dir = best_experiment / "prediction"

            test_split, experiments_roc_curve[sub_experiment_path.name] = compute_model_metrics(
                best_exp_prediction_dir, label, prediction_name
            )
            list_tests_df.append(test_split)
        else:
            raise FileExistsError(f"Best experiment  does not exist: {best_experiment}")
    tests_df = pd.concat(list_tests_df)

    experiments_metrics_dfs: Dict[str, pd.DataFrame] = {}
    for metric_name in metrics_names:
        metric_eval_df = pd.DataFrame(experiments_metrics[metric_name])
        model_eval_metrics = compute_average_metric_from_df(metric_name, metric_eval_df)

        experiments_metrics_dfs[metric_name] = combine_model_cs_eval_per_metric(
            metric_name, comparison_columns, model_eval_metrics, cs_evaluations
        )
        pd_pivot_table(experiments_metrics_dfs[metric_name])
        experiments_metrics_dfs[metric_name].to_csv(
            experiment_path / "results" / f"{metric_name}.csv", index=False
        )

    display_metrics = [
        metric_name
        for metric_name in eval_params["metrics"]
        if metric_name in experiments_metrics_dfs.keys()
    ]

    for metric_name in display_metrics:
        log.info("%s evaluation: ", metric_name)
        for line in (
            pd_pivot_table(experiments_metrics_dfs[metric_name]).to_string(index=False).split("\n")
        ):
            log.info("%s", line)

    plot_roc_curve(experiments_roc_curve, cs_evaluations, comparison_columns, experiment_path)
    plot_summary(experiments_metrics_dfs, comparison_columns, experiment_path)
    global_evaluation(
        tests_df, comparison_columns, prediction_name, label, experiment_path / "plots"
    )


def compute_average_metric_from_df(metric_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute the average of the specified metric from the experiments metrics.

    Args:
    metric_name (str): The name of the metric to compute the average for.
    df (pd.DataFrame): The DataFrame containing the experiments' metrics.

    Returns:
    pd.DataFrame: The DataFrame with the computed average metric added.
    """
    mean_metric = {"metric_name": metric_name, "experiment_name": "mean"}
    for data_name in df.drop(["metric_name", "experiment_name"], axis=1).columns:
        mean_metric[data_name] = df[data_name].mean()
    df = pd.concat([df, pd.DataFrame([mean_metric])], ignore_index=True, axis=0).reset_index(
        drop=True
    )
    return df


def compute_comparison_scores_metrics_per_split(
    comparison_columns: List[str], label: str, train_path: str, test_path: str
) -> CsEvalType:
    """Compute comparison scores metrics per split.

    Args:
        comparison_columns (List[str]): List of comparison columns.
        label (str): Label for comparison.
        train_path (str): Path to the training data.
        test_path (str): Path to the testing data.

    Returns:
        Dict[str,Dict[str,Any]]: Dictionary containing evaluation metrics for each
        comparison column.
    """
    train = read_data(train_path)
    test = read_data(test_path)
    evaluation: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for comparison_column in comparison_columns:

        train_scores = compute_scores(train, label, comparison_column)
        test_scores = compute_scores(test, label, comparison_column)

        evaluation[comparison_column]["train"] = train_scores
        evaluation[comparison_column]["validation"] = train_scores
        evaluation[comparison_column]["test"] = test_scores
    return evaluation


def combine_model_cs_eval_per_metric(
    metric_name: str,
    comparison_columns: List[str],
    model_eval_metric: pd.DataFrame,
    cs_evaluations: Dict[str, CsEvalType],
) -> pd.DataFrame:
    """Combine model and comparison score evaluation per metric.

    Args:
        metric_name (str): Name of the metric.
        comparison_columns (List[str]): List of columns for comparison.
        model_eval_metric (pd.DataFrame): DataFrame of model evaluation metrics.
        cs_evaluations (Dict[str,CsEvalType]): Dictionary of comparison score evaluations.

    Returns:
        pd.DataFrame: Combined DataFrame of evaluation metric.
    """
    list_eval_dfs: List[pd.DataFrame] = []
    model_eval_metric["model"] = "Immunogenicity Model"
    if metric_name not in ["topk", "roc", "logloss", "roc_auc_curve"]:
        return model_eval_metric

    for comparison_column in comparison_columns:
        splits_name = list(cs_evaluations.keys())
        train_scores = [
            cs_evaluations[split][comparison_column]["train"][metric_name] for split in splits_name
        ]
        val_scores = [
            cs_evaluations[split][comparison_column]["validation"][metric_name]
            for split in splits_name
        ]
        test_scores = [
            cs_evaluations[split][comparison_column]["test"][metric_name] for split in splits_name
        ]
        train_scores.append(np.mean(train_scores))
        val_scores.append(np.mean(val_scores))
        test_scores.append(np.mean(test_scores))
        splits_name.append("mean")
        cs_eval_df = pd.DataFrame(
            {
                "metric_name": [metric_name] * len(splits_name),
                "experiment_name": splits_name,
                "train": train_scores,
                "validation": val_scores,
                "test": test_scores,
            }
        )
        cs_eval_df["model"] = comparison_column
        list_eval_dfs.append(cs_eval_df)

    list_eval_dfs.append(model_eval_metric)
    return pd.concat(list_eval_dfs).reset_index(drop=True)


def pd_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pivot table on a given datafrme."""
    return pd.pivot_table(
        df,
        values=["train", "validation", "test"],
        index=["metric_name", "experiment_name"],
        columns=["model"],
    ).reset_index()


def compute_model_metrics(
    best_exp_prediction_dir: Path, label: str, prediction_name: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute the ROC curve for the model prediction.

    Args:
        best_exp_prediction_dir (Path): The directory containing the best experiment
        prediction files.
        label (str): The label to be used for evaluation.
        prediction_name (str): The name of the prediction to be used for evaluation.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation scores for the train, validation,
        and test datasets.
    """
    evaluation = {}
    if best_exp_prediction_dir.exists():
        train = pd.read_csv(best_exp_prediction_dir / "train.csv")
        test = pd.read_csv(best_exp_prediction_dir / "test.csv")
        train_scores = {"roc_auc_curve": roc_auc_curve(train[label], train["prediction"])}

        test_scores = {"roc_auc_curve": roc_auc_curve(test[label], test[prediction_name])}

        evaluation["train"] = train_scores
        evaluation["validation"] = train_scores
        evaluation["test"] = test_scores

    return test, evaluation


def compute_scores(data: pd.DataFrame, label: str, prediction_name: str) -> Dict[str, Any]:
    """Compute scores based on the input data.

    Args:
        data: A pandas DataFrame containing the data.
        label: A string representing the label column in the data.
        prediction_name: A string representing the column containing the prediction values.

    Return:
        A dictionary with keys "topk", "roc", "logloss", and "roc_auc_curve" and their respective
        computed scores.
    """
    return {
        "topk": topk(data[label], data[prediction_name]),
        "roc": roc_auc_score(data[label], data[prediction_name]),
        "logloss": logloss(data[label], data[prediction_name]),
        "roc_auc_curve": roc_auc_curve(data[label], data[prediction_name]),
    }


def plot_roc_curve(
    experiments_roc_curve: Dict[str, Dict[str, Any]],
    cs_evaluations: Dict[str, CsEvalType],
    comparison_columns: List[str],
    experiments_dr: Path,
) -> None:
    """Plot ROC curves for different experiments and save the plots in the specified directory.

    Args:
        experiments_roc_curve (Dict[str, Dict[str, Any]]): A dictionary containing ROC curves for
        different experiments.
        cs_evaluations (Dict[str, CsEvalType]): A dictionary containing evaluations for different
        experiments.
        comparison_columns (List[str]): A list of columns for comparison.
        experiments_dr (Path): The directory where the plots will be saved.

    Returns:
        None
    """
    plots_dr = experiments_dr / "plots"
    plots_dr.mkdir(exist_ok=True, parents=True)

    for sub_exp in cs_evaluations.keys():
        for partition in ["train", "validation", "test"]:
            _, ax = plt.subplots(figsize=(8, 8), ncols=1)
            for comparison_column in comparison_columns:
                cs_fpr, cs_tpr = cs_evaluations[sub_exp][comparison_column][partition][
                    "roc_auc_curve"
                ]
                ax.plot(cs_fpr, cs_tpr, label=f"{comparison_column}")
            ax.plot([0, 1], [0, 1], color="green", lw=2, linestyle="--", label="random")
            model_fpr, model_tpr = experiments_roc_curve[sub_exp][partition]["roc_auc_curve"]
            ax.plot(model_fpr, model_tpr, label="Immunogenicity Model")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            ax.set_title(f"AUC ROC curve plot {partition}:{sub_exp} ")
            plt.savefig(plots_dr / f"{sub_exp}_{partition}_roc_auc_curve.png")
            plt.clf()
            plt.close()


def plot_summary(
    experiments_metrics_dfs: Dict[str, pd.DataFrame],
    comparison_columns: List[str],
    experiments_dr: Path,
) -> None:
    """Plot summary of experiments metrics for comparison columns and save the plots.

    Args:
        experiments_metrics_dfs (Dict[str, pd.DataFrame]): Dictionary of experiment metrics
        dataframes
        comparison_columns (List[str]): List of comparison columns
        experiments_dr (Path): Path to experiments directory

    Returns:
        None
    """
    plot_title = {"topk": "Topk", "roc": "AUC-ROC"}
    for metric in ["topk", "roc"]:
        experiments_metric = experiments_metrics_dfs[metric]
        for comparison_column in comparison_columns:
            fig, ax = plt.subplots(figsize=(24, 8), ncols=3)
            palette_colors = {
                "Immunogenicity Model": mcolors.CSS4_COLORS["darkgreen"],
                comparison_column: mcolors.CSS4_COLORS["limegreen"],
            }
            other_comparison_columns = [
                col for col in comparison_columns if col != comparison_column
            ]
            df = experiments_metric[~experiments_metric.model.isin(other_comparison_columns)]
            for i, partition in enumerate(["train", "validation", "test"]):
                plot_bar_plot(df, comparison_column, ax[i], palette_colors, partition, metric)

            fig.suptitle(
                f"{plot_title[metric]} Bar plot: Immunogenicity Model VS {comparison_column}",
                fontsize=20,
            )
            fig.tight_layout()
            fig.savefig(
                experiments_dr / "plots" / f"{metric}_IgVs{comparison_column}_ScoreSummary.png"
            )
            plt.close("all")


def plot_bar_plot(
    data: pd.DataFrame,
    comparison_column: str,
    ax: matplotlib.axes.SubplotBase,
    palette_colors: Dict[str, Any],
    partition: str,
    metric_name: str,
) -> None:
    """Plot a bar plot for the given data.

    Using the specified comparison column,axis, colors partition, and metric name.
    Args:
        data (pd.DataFrame): The input data frame.
        comparison_column (str): The column for comparison.
        ax (matplotlib.axes.SubplotBase): The axis to plot on.
        palette_colors (Dict[str, Any]): The palette colors for the plot.
        partition (str): The partition to plot.
        metric_name (str): The name of the metric.

    Returns:
        None
    """
    sns.barplot(
        data=data,
        x="experiment_name",
        y=partition,
        hue="model",
        palette=palette_colors,
        hue_order=["Immunogenicity Model", comparison_column],
        ax=ax,
    )

    ax.set_title(f"{partition}", fontsize=15)
    ax.set_xlabel("split", fontsize=15)
    ax.set_ylabel(metric_name, fontsize=15)
    ax.legend(loc="lower left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%0.3f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


def global_evaluation(
    tests_df: pd.DataFrame,
    comparison_columns: List[str],
    prediction_name: str,
    label: str,
    plots_dr: Path,
) -> None:
    """Perform global evaluation on the given test data and save the plots.

    Args:
        tests_df (pd.DataFrame): The DataFrame containing the test data.
        comparison_columns (List[str]): The list of columns to be used for comparison.
        prediction_name (str): The name of the prediction column.
        label (str): The name of the label column.
        plots_dr (Path): The directory to save the plots.

    Returns:
        None
    """
    scores = {}
    for comparison_column in comparison_columns:
        scores[comparison_column] = compute_scores(tests_df, label, comparison_column)

    scores["Immunogenicity Model"] = compute_scores(tests_df, label, prediction_name)
    log.info("Evaluate Public Table (Combined test splits):")
    for metric_name in ["topk", "roc", "logloss"]:
        log.info(metric_name)
        for model_name in scores:
            log.info(f"  - {model_name}: {scores[model_name][metric_name]:.3}")

    _, ax = plt.subplots(figsize=(8, 8), ncols=1)
    for model_name in scores:
        fpr, tpr = scores[model_name]["roc_auc_curve"]
        ax.plot(fpr, tpr, label=f"{model_name}")

    ax.plot([0, 1], [0, 1], color="green", lw=2, linestyle="--", label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.set_title("AUC ROC curve plot Public table (Combined test splits)")
    plt.savefig(plots_dr / "Global_roc_auc_curve.png")
    plt.clf()
    plt.close()
