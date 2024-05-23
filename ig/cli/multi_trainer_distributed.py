"""Module used to train many experiments with one configuration file."""
import copy
import json
import os
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import click
import pandas as pd

from ig import CONFIGURATION_DIRECTORY, DEFAULT_SEED, MODELS_DIRECTORY
from ig.bucket.click import arguments
from ig.cli.trainer import _check_model_folder, train
from ig.src.logger import get_logger, init_logger
from ig.src.utils import load_yml, save_yml, seed_basic

log: Logger = get_logger("Multi_train")
seed_basic(DEFAULT_SEED)


@click.command()
@click.option(
    "--train_data_path",
    "-train",
    type=str,
    required=True,
    help="Path to the dataset used in training",
)
@click.option(
    "--test_data_path",
    "-test",
    type=str,
    required=True,
    help="Path to the dataset used in  evaluation.",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
@click.option(
    "--default_configuration_file",
    "-dc",
    type=str,
    required=True,
    help=" Path to the deafult configuration file.",
)
@arguments.force(help="Force overwrite the local file if it already exists.")  # type: ignore
@click.pass_context
def multi_train_distributed(
    ctx: click.core.Context,
    train_data_path: str,
    test_data_path: str,
    configuration_file: str,
    default_configuration_file: str,
    folder_name: str,
    force: bool,
) -> None:
    """Launch the traning of multiple experimnets using one configuration file."""
    experiments_path = MODELS_DIRECTORY / folder_name
    experiments_configuration_path = experiments_path / "configuration"
    experiment_results_path = experiments_path / "results"
    _check_model_folder(experiments_path)
    experiments_configuration_path.mkdir(exist_ok=True, parents=True)
    experiment_results_path.mkdir(exist_ok=True, parents=True)
    init_logger(folder_name)
    log.info("Started")
    default_configuration = load_yml(CONFIGURATION_DIRECTORY / default_configuration_file)
    configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    experiments_configuration = configuration["experiments"]
    params = configuration["params"]
    tf_config = json.loads(os.environ.get("TF_CONFIG") or "{}")
    task_config = tf_config.get("task", {})
    task_index = task_config.get("index")

    def train_fun(experiment_name: str) -> None:
        """Main train function to run on experiments."""
        log.info("%s is running", experiment_name)
        experiment_path = experiments_path / experiment_name
        experiment_configuration_path = generate_experiment_configuration(
            default_configuration=default_configuration,
            experiment_configuration=experiments_configuration[experiment_name],
            experiments_configuration_path=experiments_configuration_path,
            experiment_name=experiment_name,
        )
        ctx.invoke(
            train,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            unlabeled_path=None,
            configuration_file=experiment_configuration_path,
            folder_name=experiment_path,
            force=force,
            multi_train=True,
        )
        experiment_paths.append(experiment_path)

    experiment_paths: List[Path] = []
    for i, experiment_name in enumerate(experiments_configuration.keys()):
        if task_index == i:
            train_fun(experiment_name=experiment_name)
    experiments_evaluation(experiment_paths, experiments_path, params)


def experiments_evaluation(
    experiment_paths: List[Path], experiments_path: Path, params: Dict[str, Any]
) -> None:
    """Evalute each experimnet and save the results."""
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
        log.info("%s evalution: ############################################", metric_name)
        for line in experiments_metrics_dfs[metric_name].to_string().split("\n"):
            log.info("%s", line)


def generate_experiment_configuration(
    default_configuration: Dict[str, Any],
    experiment_configuration: Dict[str, Any],
    experiments_configuration_path: Path,
    experiment_name: str,
) -> str:
    """Make a copy and modify the default configuration file.

    Change the default configuration settings with the new settings from experiment configuration
    save the new configuration file in a temporary folder.

    Args:
        default_configuration: Dictionary contains the default settings
        experiment_configuration: Dictionary holds the new settings
        experiments_configuration_path: Path where the generted configuration files will bes saved
        experiment_name: name of the experiment

    Output:
        path to the modified configuration file
    """
    configuration = copy.deepcopy(default_configuration)
    for key in experiment_configuration.keys():
        if isinstance(experiment_configuration[key], Dict):
            for second_key in experiment_configuration[key].keys():
                key_configuration = configuration[key]
                key_configuration[second_key] = experiment_configuration[key][second_key]
            configuration[key] = key_configuration
        else:
            configuration[key] = experiment_configuration[key]

    path = str(experiments_configuration_path / f"{experiment_name}.yml")
    save_yml(configuration, path)
    return path
