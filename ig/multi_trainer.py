"""Module used to train many experiments with one configuration file."""
import copy
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Union

import click
import pandas as pd

from ig import CONFIGURATION_DIRECTORY, DEFAULT_SEED, MODELS_DIRECTORY
from ig.bucket.click import arguments
from ig.inference import exp_inference, inference
from ig.src.logger import get_logger, init_logger
from ig.src.metrics import global_evaluation
from ig.src.utils import load_yml, save_yml, seed_basic
from ig.trainer import _check_model_folder, train

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
def multi_train(
    ctx: Union[click.core.Context, Any],
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
    default_configuration = load_yml(CONFIGURATION_DIRECTORY / default_configuration_file)
    configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    experiments_configuration = configuration["experiments"]
    params = configuration["params"]
    save_yml(configuration, experiments_configuration_path / "multi-train-configuration.yml")
    cli_train_data_path = train_data_path
    cli_test_data_path = test_data_path
    experiment_paths: List[Path] = []
    init_logger(
        logging_directory=experiments_path,
        file_name="multi-train",
    )
    log.info("Start")
    for experiment_name in experiments_configuration.keys():
        log.info("The %s experiment is running", experiment_name)
        experiment_path = experiments_path / experiment_name
        experiment_configuration_path = generate_experiment_configuration(
            default_configuration=default_configuration,
            experiment_configuration=experiments_configuration[experiment_name],
            experiments_configuration_path=experiments_configuration_path,
            experiment_name=experiment_name,
        )
        dataset = experiments_configuration[experiment_name].get("dataset", None)
        if dataset:
            train_data_path = dataset["train"]
            test_data_path = dataset["test"]
        else:
            train_data_path = cli_train_data_path
            test_data_path = cli_test_data_path

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
        init_logger(
            logging_directory=experiments_path,
            file_name="multi-train",
        )
    log.info("Multi-experiment results")

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
        log.info("%s evaluation: ", metric_name)
        for line in experiments_metrics_dfs[metric_name].to_string(index=False).split("\n"):
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
        if isinstance(experiment_configuration[key], Dict) and key != "dataset":
            for second_key in experiment_configuration[key].keys():
                key_configuration = configuration[key]
                key_configuration[second_key] = experiment_configuration[key][second_key]
            configuration[key] = key_configuration
        else:
            configuration[key] = experiment_configuration[key]

    path = str(experiments_configuration_path / f"{experiment_name}.yml")
    save_yml(configuration, path)
    return path


@click.command()
@click.option(
    "--test_data_paths",
    "-d",
    type=str,
    required=True,
    help="Path of the dataset.",
    multiple=True,
)
@click.option(
    "--single_train_names",
    "-sn",
    type=str,
    help="run's names under the multi train run's name directory.",
    multiple=True,
    required=False,
)
@click.option(
    "--multi_train_run_name",
    "-mn",
    type=str,
    required=True,
    help="multi train run's name directory",
)
@click.option(
    "--id_name", "-id", type=str, required=True, help="unique id for the data set", default="id"
)
@click.option("--label_name", "-l", type=str, help="label name", default=None)
@click.option("--process", "-p", is_flag=True, help="process and clean the provided data set")
@click.option("--eval_test", "-e", is_flag=True, help="whether eval and compute metrics or not")
@click.pass_context
def multi_inference(
    ctx: Union[click.core.Context, Any],
    test_data_paths: List[str],
    multi_train_run_name: str,
    id_name: str,
    process: bool,
    label_name: Optional[str],
    eval_test: bool,
    single_train_names: Optional[List[str]] = None,
) -> None:
    """Inference command for Multi-training run."""
    predications_eval: Dict[str, List[pd.Series]] = defaultdict(list)

    multi_train_directory = MODELS_DIRECTORY / multi_train_run_name
    init_logger(logging_directory=multi_train_directory, file_name="Inference")
    log.info("Start Inference.")

    multi_train_configuration_directory = multi_train_directory / "configuration"
    multi_train_results_directory = multi_train_directory / "results"
    multi_train_configuration = load_yml(
        multi_train_configuration_directory / "multi-train-configuration.yml"
    )
    single_train_configurations = multi_train_configuration["experiments"]
    if not single_train_names:
        single_train_names = list(single_train_configurations.keys())
    log.info("the Inference will be done for the following runs:")
    log.info("%s", "|".join(single_train_names))
    for test_data_path in test_data_paths:
        log.info("Working on %s", test_data_path)
        for single_train_name in single_train_names:
            log.info("Run : %s", single_train_name)
            single_train_directory = multi_train_directory / single_train_name
            ctx.invoke(
                inference,
                test_data_path=test_data_path,
                folder_name=single_train_directory,
                id_name=id_name,
                process=process,
            )
            if eval_test and (label_name is not None):
                eval_test_data(
                    test_data_path,
                    single_train_directory,
                    single_train_name,
                    label_name,
                    predications_eval,
                )
            init_logger(logging_directory=multi_train_directory, file_name="Inference")

    if len(predications_eval) > 0:
        for key in predications_eval:
            log.info("%s evalution :", key)
            metrics_df = pd.DataFrame(predications_eval[key])
            for line in metrics_df.to_string(index=False).split("\n"):
                log.info("%s", line)
            metrics_df.to_csv(multi_train_results_directory / f"scores_{key}.csv", index=False)


def eval_test_data(
    test_data_path: str,
    single_train_directory: Path,
    single_train_name: str,
    label_name: str,
    predications_eval: Dict[str, List[pd.Series]],
) -> None:
    """Compute metrics for a given file after loading it's predication."""
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


@click.command()
@click.option(
    "--test_data_paths",
    "-d",
    type=str,
    required=True,
    help="Path of the dataset.",
    multiple=True,
)
@click.option(
    "--multi_train_run_name",
    "-mn",
    type=str,
    required=True,
    help="multi train run's name directory",
)
@click.option("--process", "-p", is_flag=True, help="process and clean the provided data set")
@click.pass_context
def multi_exp_inference(
    ctx: Union[click.core.Context, Any],
    test_data_paths: List[str],
    multi_train_run_name: str,
    process: bool,
) -> None:
    """Inference command for Multi-training run."""
    multi_train_directory = MODELS_DIRECTORY / multi_train_run_name
    init_logger(logging_directory=multi_train_directory, file_name="Exp-Inference")
    log.info("Start Inference.")

    multi_train_configuration_directory = multi_train_directory / "configuration"
    multi_train_configuration = load_yml(
        multi_train_configuration_directory / "multi-train-configuration.yml"
    )
    single_train_configurations = multi_train_configuration["experiments"]

    single_train_names = list(single_train_configurations.keys())
    log.info("the Inference will be done for the following runs:")
    log.info("%s", "|".join(single_train_names))
    for test_data_path in test_data_paths:
        log.info("Working on %s", test_data_path)
        for single_train_name in single_train_names:
            log.info("Run : %s", single_train_name)
            single_train_directory = multi_train_directory / single_train_name
            ctx.invoke(
                exp_inference,
                test_data_path=test_data_path,
                run_name=single_train_directory,
                process=process,
            )
            init_logger(logging_directory=multi_train_directory, file_name="Exp-Inference")
