"""Module used to define some helper functions for the trainer script."""
import copy
import shutil
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import pandas as pd

import ig.cross_validation as exper
from ig import (
    DATAPROC_DIRECTORY,
    EXP_NAMES,
    EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY,
    FEATURES_DIRECTORY,
    FEATURES_SELECTION_DIRECTORY,
    KFOLD_MULTISEED_MODEL_NAME,
    MODEL_CONFIGURATION_DIRECTORY,
)
from ig.constants import EvalExpType
from ig.dataset.dataset import Dataset
from ig.utils.cross_validation import import_experiment
from ig.utils.general import generate_random_seeds, log
from ig.utils.logger import NeptuneLogs


def generate_single_exp_config(
    configuration: Dict[str, Any],
    experiment_name: str,
    experiment_param: Dict[str, Any],
    model_type: str,
    model_param: Dict[str, Any],
) -> Dict[str, Union[str, Dict[str, str]]]:
    """Cleanup the configuration file for each Experiment."""
    configuration = copy.deepcopy(configuration)
    configuration.pop("feature_paths", None)
    configuration["processing"].pop("folds", None)
    configuration["processing"].pop("seeds", None)
    if experiment_name == KFOLD_MULTISEED_MODEL_NAME:
        if not experiment_param.get("seeds", None):
            experiment_param["seeds"] = generate_random_seeds(
                nbr_seeds=experiment_param["nbr_seeds"]
            )
    configuration["experiments"] = {experiment_name: experiment_param}
    configuration["model_type"] = model_type
    configuration["model"] = model_param

    return configuration


def train_func(  # noqa: CCR001
    experiment_name: str,
    experiment_param: Dict[str, Any],
    model_type: str,
    model_param: Dict[str, Any],
    train_data: Dataset,
    test_data: Dataset,
    unlabeled_path: str,
    configuration: Dict[str, Any],
    folder_name: str,
    log_handler: Logger,
    neptune_log: Optional[NeptuneLogs] = None,
    sub_folder_name: Optional[str] = None,
    comparison_score_metrics: Optional[pd.DataFrame] = None,
) -> Tuple[List[EvalExpType], str]:
    """Train the same experiment with different features lists."""
    display = ""

    experiment_configuration = generate_single_exp_config(
        configuration=configuration,
        experiment_name=experiment_name,
        experiment_param=experiment_param,
        model_type=model_type,
        model_param=model_param,
    )

    experiment_class = import_experiment(exper, experiment_name)
    features_list_paths = copy.deepcopy(configuration["feature_paths"])
    eval_configuration = configuration["evaluation"]
    results = []
    display += f"\n -{model_type} : \n"
    for features_list_path in features_list_paths:
        sub_model_features = features_list_path
        log_handler.info("          -features set :%s", sub_model_features)
        features_sub_folder_name = (
            f"{sub_model_features}/{sub_folder_name}" if sub_folder_name else sub_model_features
        )

        experiment_configuration["features"] = features_list_path
        experiment = experiment_class(
            train_data=train_data,
            test_data=test_data,
            unlabeled_path=unlabeled_path,
            configuration=experiment_configuration,
            experiment_name=experiment_name,
            folder_name=folder_name,
            sub_folder_name=features_sub_folder_name,
            features_configuration_path=train_data.features_configuration_path,
            **experiment_param,
        )

        experiment.train()
        scores = experiment.eval_exp(comparison_score_metrics=comparison_score_metrics)

        validation_score = scores["validation"][eval_configuration["metric_selector"]].iloc[0]
        if not validation_score:
            validation_score = ""
        test_score = scores["test"][eval_configuration["metric_selector"]].iloc[0]
        prediction_name = scores["validation"]["prediction"].iloc[0]

        display += f"  * Features: {sub_model_features}\n"
        display += (
            f"     {prediction_name} :"
            f" Validation : {validation_score:0.3f}"
            f" Test:{test_score:0.3f}\n"
        )

        results.append(scores)
        if neptune_log:
            neptune_log.upload_experiment(experiment_path=experiment.experiment_directory)
    return results, display


def load_datasets(
    click_ctx: Union[click.core.Context, Any],
    train_data_path: str,
    test_data_path: str,
    general_configuration: Dict[str, Any],
    experiment_path: Path,
    force: bool,
) -> Tuple[Dataset, Dataset]:
    """Load train , test data for a given paths."""
    train_data = Dataset(
        click_ctx=click_ctx,
        data_path=train_data_path,
        configuration=general_configuration,
        is_train=True,
        experiment_path=experiment_path,
        force_gcp=force,
    ).load_data()
    test_data = Dataset(
        click_ctx=click_ctx,
        data_path=test_data_path,
        configuration=general_configuration,
        is_train=False,
        experiment_path=experiment_path,
        force_gcp=force,
        is_inference=True,
        process_label=True,
    ).load_data()
    return train_data, test_data


def get_best_experiment(
    results: pd.DataFrame,
    eval_configuration: Dict[str, Any],
    path: Path,
    file_name: str = "results",
) -> Tuple[str, str]:
    """Return the best experiment."""
    display = ""
    metric_selector = eval_configuration.get("metric_selector", "topk")
    monitoring_metrics = eval_configuration.get("monitoring_metrics", [metric_selector])
    validation = pd.concat([result["validation"] for result in results])
    test = pd.concat([result["test"] for result in results])
    pd.concat([validation, test]).to_csv(path / f"{file_name}.csv", index=False)
    statistic_list = []
    for result in results:
        try:
            statistic_list.append(result["statistic"])
        except KeyError:
            pass
    data_selector = pd.concat(
        [result[eval_configuration["data_name_selector"]] for result in results]
    )
    data_selector.sort_values(
        [metric_selector], ascending=eval_configuration["metrics_selector_higher"], inplace=True
    )

    best_experiment_id = data_selector.iloc[-1]["ID"]

    best_validation = validation[validation.ID == best_experiment_id]
    best_test = test[test.ID == best_experiment_id]

    display += "###### Best run ######\n"
    display += f"Experiment : {best_test.experiment.iloc[0]} \n"
    display += f"model :  {best_test.model.iloc[0]}\n"
    display += f"features :  {best_test.features.iloc[0]}\n"
    display += f"seed :  {best_test.seed.iloc[0]}\n" if "seed" in best_test.columns else ""
    display += (
        f"nbr of folds :  {best_test.nbr_fold.iloc[0]}\n" if "nbr_fold" in best_test.columns else ""
    )
    display += f"prediction :  {best_test.prediction.iloc[0]}\n"
    validation_metrics_display = " ".join(
        [f"{metric} : {best_validation[metric].iloc[0]:.3f}" for metric in monitoring_metrics]
    )
    display += f"Validation score :  {validation_metrics_display}\n"
    test_metrics_display = " ".join(
        [f"{metric} : {best_test[metric].iloc[0]:.3f}" for metric in monitoring_metrics]
    )
    display += f"Test score :  {test_metrics_display}\n"
    if len(statistic_list) > 0:
        statistic = pd.concat(statistic_list)
        beststatistic_list = statistic[statistic.ID == best_experiment_id]
        if len(beststatistic_list):
            for e in beststatistic_list.drop(["ID"], axis=1).columns:
                display += f"{e} :  {beststatistic_list[e].iloc[0]}\n"
    return best_experiment_id, display


def get_task_name(label_name: str) -> str:
    """Return the folder name to the corresponding label name."""
    if label_name.lower() == "cd4_any":
        return "CD4"
    if label_name.lower() == "cd8_any":
        return "CD8"
    raise ValueError("label name not known in fs ...")


def log_summary_results(display: str) -> None:
    """Print summary results."""
    for line in display.split("\n"):
        log.info(line)


def copy_existing_features_lists(
    feature_list: List[str], experiment_path: Path, label_name: str
) -> None:
    """Move features list from features directory to the experiment directory."""
    if feature_list:
        original_directory = FEATURES_DIRECTORY / get_task_name(label_name=label_name)
        features_selection_directory = experiment_path / FEATURES_SELECTION_DIRECTORY
        features_selection_directory.mkdir(exist_ok=True, parents=True)
        for feature in feature_list:
            shutil.copyfile(
                original_directory / f"{feature}.txt",
                experiment_path / FEATURES_SELECTION_DIRECTORY / f"{feature}.txt",
            )


def copy_models_configuration_files(
    general_configuration: Dict[str, Any], experiment_path: Path
) -> None:
    """Copy model configuration file to  the experiment folder."""
    models_configuration_directory = experiment_path / EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY
    models_configuration_directory.mkdir(exist_ok=True, parents=True)
    for models_configuration_name in general_configuration["models"]:
        shutil.copyfile(
            MODEL_CONFIGURATION_DIRECTORY / models_configuration_name,
            models_configuration_directory / models_configuration_name,
        )


def remove_unnecessary_folders(experiment_path: Path, remove: bool) -> None:
    """Remove unnecessary folders inside the experiment directory."""
    subfolders_to_remove = EXP_NAMES + [DATAPROC_DIRECTORY]
    for directory in subfolders_to_remove:
        path = experiment_path / directory

        if remove and path.exists():

            if directory == DATAPROC_DIRECTORY:
                data_processing_files = path.iterdir()
                files_to_remove = list(
                    filter(
                        lambda x: x.suffix in [".csv", ".tsv", ".xlsx"], list(data_processing_files)
                    )
                )
                for file in files_to_remove:

                    file.unlink()
            else:
                shutil.rmtree(path)


def save_best_experiment(best_experiment_id: List[str], experiment_path: Path) -> None:
    """Save best experiment."""
    if len(best_experiment_id) == 4:
        ## TODO change the saving folder order to match 0,1,2,3 instead of 0,1,3,2
        path = (
            experiment_path
            / best_experiment_id[0]
            / best_experiment_id[1]
            / best_experiment_id[3]
            / best_experiment_id[2]
        )
    else:
        path = (
            experiment_path / best_experiment_id[0] / best_experiment_id[1] / best_experiment_id[2]
        )
    best_experiment_path = path
    destination_path = experiment_path / "best_experiment"
    shutil.copytree(src=best_experiment_path, dst=destination_path, dirs_exist_ok=True)
