"""Module used to train many experiments with one configuration file."""
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import pandas as pd

from ig import CONFIGURATION_DIRECTORY, DEFAULT_SEED, MODELS_DIRECTORY
from ig.bucket.click import arguments
from ig.cli.inference import exp_inference, inference
from ig.cli.trainer import train
from ig.utils.evaluation import eval_test_data, experiments_evaluation
from ig.utils.general import seed_basic
from ig.utils.io import check_model_folder, generate_experiment_configuration, load_yml, save_yml
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
    check_model_folder(experiments_path)
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
@click.option("--eval_test", "-e", is_flag=True, help="whether eval and compute metrics or not")
@click.pass_context
def multi_inference(
    ctx: Union[click.core.Context, Any],
    test_data_paths: List[str],
    multi_train_run_name: str,
    id_name: str,
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
@click.pass_context
def multi_exp_inference(
    ctx: Union[click.core.Context, Any],
    test_data_paths: List[str],
    multi_train_run_name: str,
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
            )
            init_logger(logging_directory=multi_train_directory, file_name="Exp-Inference")
