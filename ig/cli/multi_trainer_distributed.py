"""Module used to train many experiments with one configuration file."""
import json
import os
from logging import Logger
from pathlib import Path
from typing import List

import click

from ig import CONFIGURATION_DIRECTORY, DEFAULT_SEED, MODELS_DIRECTORY
from ig.bucket.click import arguments
from ig.cli.trainer import train
from ig.utils.evaluation import experiments_evaluation
from ig.utils.general import seed_basic
from ig.utils.io import check_model_folder, generate_experiment_configuration, load_yml
from ig.utils.logger import get_logger, init_logger

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
    help=" Path to the default configuration file.",
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
    """Launch the training of multiple experiments using one configuration file."""
    experiments_path = MODELS_DIRECTORY / folder_name
    experiments_configuration_path = experiments_path / "configuration"
    experiment_results_path = experiments_path / "results"
    check_model_folder(experiments_path)
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
