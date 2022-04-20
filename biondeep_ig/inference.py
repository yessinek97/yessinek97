"""Module used to define the inference command on a new test set ."""
from pathlib import Path

import click

import biondeep_ig.src.experiments as exper
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.processing_v1 import Dataset
from biondeep_ig.src.utils import import_experiment
from biondeep_ig.src.utils import load_yml

log = get_logger("Inference")


@click.command()
@click.option(
    "--test_data_path",
    "-d",
    type=str,
    required=True,
    help="Path to the dataset.",
)
@click.option(
    "--id_name",
    "-id",
    type=str,
    required=True,
    help="unique id for the data set",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
def inference(test_data_path, folder_name, id_name):
    """Inferring Method.

    Args:
        test_data_path: test path
        folder_name: checkpoint name
        id_name: the unique id for each row in teh data set
    """
    init_logger(folder_name, "Inference")

    best_exp_path = MODELS_DIRECTORY / folder_name / "best_experiment"

    exp_configuration = load_yml(best_exp_path / "configuration.yml")
    experiment_name = list(exp_configuration["experiments"])[0]
    experiment_params = exp_configuration["experiments"][experiment_name]
    features_file_path = best_exp_path / "features.txt"

    prediction_path = MODELS_DIRECTORY / "Inference" / folder_name
    prediction_path.mkdir(parents=True, exist_ok=True)
    prediction_name_selector = exp_configuration["evaluation"]["prediction_name_selector"]
    experiment_class = import_experiment(exper, experiment_name)
    file_name = Path(test_data_path).stem

    log.info((f"Inferring  on {file_name}.csv " f"using best experiment from run : {folder_name}"))
    log.info(f"Experiment : {experiment_name}")
    log.info(f"Model Type : {exp_configuration['model_type']}")
    log.info(f"Features   : {exp_configuration['features']}")
    log.info(f"prediction name selector : {prediction_name_selector}")
    test_data = Dataset(
        data_path=test_data_path,
        configuration=exp_configuration,
        is_train=False,
        experiment_path=best_exp_path.parent,
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
        **experiment_params,
    )

    predictions = experiment.inference(experiment.test_data(), save_df=False)
    predictions.to_csv(prediction_path / f"{file_name}.csv", index=False)
    predictions["prediction"] = predictions[prediction_name_selector]
    predictions[[id_name.lower(), "prediction"]].to_csv(
        prediction_path / (file_name + ".csv"), index=False
    )

    log.info(f"predictions  path  {prediction_path}/{file_name}.csv")
