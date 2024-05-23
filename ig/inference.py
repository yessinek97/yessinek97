"""Module used to define the inference command on a new test set ."""
import copy
from logging import Logger
from pathlib import Path
from typing import Any, Union

import click

import ig.cross_validation as exper
from ig import DATAPROC_DIRECTORY, MODELS_DIRECTORY
from ig.dataset.dataset import Dataset
from ig.src.logger import get_logger, init_logger
from ig.src.utils import import_experiment, load_experiments, load_models, load_yml

log: Logger = get_logger("Inference")


@click.command()
@click.option(
    "--test_data_path",
    "-d",
    type=str,
    required=True,
    help="Path to the dataset.",
)
@click.option(
    "--id_name", "-id", type=str, required=True, help="unique id for the data set", default="id"
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.pass_context
def inference(
    ctx: Union[click.core.Context, Any],
    test_data_path: str,
    folder_name: Union[str, Path],
    id_name: str,
) -> None:
    """Inferring Method.

    Args:
        ctx: Click context manager
        test_data_path: test path
        folder_name: checkpoint name
        id_name: the unique id for each row in the data set
    """
    if isinstance(folder_name, str):
        best_exp_path = MODELS_DIRECTORY / folder_name / "best_experiment"
    if isinstance(folder_name, Path):
        best_exp_path = folder_name / "best_experiment"
    exp_configuration = load_yml(best_exp_path / "configuration.yml")
    experiment_name = list(exp_configuration["experiments"])[0]
    experiment_params = exp_configuration["experiments"][experiment_name]
    features_file_path = best_exp_path / "features.txt"

    prediction_path = best_exp_path.parent / "Inference"
    prediction_path.mkdir(parents=True, exist_ok=True)
    features_configuration_path = (
        best_exp_path.parent / DATAPROC_DIRECTORY / "features_configuration.yml"
    )
    init_logger(prediction_path, "Inference")
    prediction_name_selector = exp_configuration["evaluation"]["prediction_name_selector"]
    experiment_class = import_experiment(exper, experiment_name)
    file_name = Path(test_data_path).stem

    log.info("Inferring  on  %s using best experiment from run %s:", file_name, folder_name)
    log.info("Experiment : %s", experiment_name)
    log.info("Model Type : %s", exp_configuration["model_type"])
    log.info("Features   : %s", exp_configuration["features"])
    log.info("prediction name selector : %s", prediction_name_selector)
    test_data = Dataset(
        click_ctx=ctx,
        data_path=test_data_path,
        configuration=exp_configuration,
        is_train=False,
        experiment_path=best_exp_path.parent,
        force_gcp=True,
        is_inference=True,
        process_label=False,
    ).load_data()
    experiment = experiment_class(
        train_data=None,
        test_data=test_data,
        configuration=exp_configuration,
        experiment_name=experiment_name,
        folder_name=folder_name,
        sub_folder_name=exp_configuration["features"],
        experiment_directory=best_exp_path,
        features_file_path=features_file_path,
        features_configuration_path=features_configuration_path,
        **experiment_params,
    )

    predictions = experiment.inference(test_data(), save_df=False)
    predictions.to_csv(prediction_path / f"{file_name}.csv", index=False)
    predictions["prediction"] = predictions[prediction_name_selector]
    predictions[[id_name.lower(), "prediction"]].to_csv(
        prediction_path / (file_name + "pred_only.csv"), index=False
    )

    log.info("predictions  path %s.csv", prediction_path / file_name)


@click.command()
@click.option(
    "--test_data_path",
    "-d",
    type=str,
    required=True,
    help="Path to the dataset.",
)
@click.option("--run_name", "-n", type=str, required=True, help="run name.")
@click.pass_context
def exp_inference(
    ctx: Union[click.core.Context, Any],
    test_data_path: str,
    run_name: Union[str, Path],
) -> None:
    """Inferring Method.

    Args:
        ctx: Click context manager
        test_data_path: test path
        run_name: checkpoint name
    """
    if isinstance(run_name, str):
        run_folder = MODELS_DIRECTORY / run_name
    if isinstance(run_name, Path):
        run_folder = run_name
    init_logger(run_folder, "Exp-Inference")
    run_configuration = load_yml(run_folder / "configuration.yml")
    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(run_configuration)
    log.info("****************************** Load EXP ****************************** ")
    model_types, model_params = load_models(run_configuration, run_folder)
    log.info("****************************** Load Models ****************************** ")

    features_list_names = copy.deepcopy(run_configuration["feature_paths"])
    log.info("****************************** Load Features lists *****************************")
    test_data = Dataset(
        click_ctx=ctx,
        data_path=test_data_path,
        configuration=run_configuration,
        is_train=False,
        experiment_path=run_folder,
        force_gcp=True,
        is_inference=True,
        process_label=False,
    ).load_data()
    file_name = Path(test_data_path).stem
    for experiment_name, experiment_param in zip(experiment_names, experiment_params):
        log.info("- %s", experiment_name)
        experiment_class = import_experiment(exper, experiment_name)
        for model_type, _ in zip(model_types, model_params):
            log.info("-- %s", model_type)
            for features_list_name in features_list_names:
                single_train_directory = (
                    run_folder / experiment_name / features_list_name / model_type
                )
                configuration = load_yml(single_train_directory / "configuration.yml")

                features_configuration_path = (
                    run_folder / DATAPROC_DIRECTORY / "features_configuration.yml"
                )

                log.info("--- %s", features_list_name)
                experiment = experiment_class(
                    train_data=None,
                    test_data=test_data,
                    configuration=configuration,
                    experiment_name=experiment_name,
                    folder_name=run_folder,
                    sub_folder_name=features_list_name,
                    features_configuration_path=features_configuration_path,
                    **experiment_param,
                )
                predictions = experiment.inference(test_data(), save_df=False)
                predictions.to_csv(
                    single_train_directory / "prediction" / f"{file_name}.csv", index=False
                )
                log.info(
                    "Saving the inference prediction at %s",
                    str(single_train_directory / "prediction" / f"{file_name}.csv"),
                )
