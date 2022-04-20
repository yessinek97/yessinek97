"""Module used to define the training command."""
import copy
import shutil
from pathlib import Path

import click

import biondeep_ig.src.feature_selection as fese
from biondeep_ig import CONFIGURATION_DIRECTORY
from biondeep_ig import FEATURES_SELECTION_DIRACTORY
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.logger import NeptuneLogs
from biondeep_ig.src.processing_v1 import Dataset
from biondeep_ig.src.utils import get_model_by_name
from biondeep_ig.src.utils import load_fs
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import save_yml

log = get_logger("FeatureSelection")


@click.command()
@click.option(
    "--train_data_path",
    "-train",
    type=str,
    required=True,
    help="Path to the dataset used in training",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
def featureselection(train_data_path, configuration_file, folder_name):
    """Feature selection process."""
    _check_model_folder(folder_name)
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    experiment_path = Path(MODELS_DIRECTORY / folder_name)

    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["feature selection"], training_path=train_data_path, test_path=None
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)

    train_data = Dataset(
        data_path=train_data_path,
        configuration=general_configuration,
        is_train=True,
        experiment_path=experiment_path,
    ).load_data()
    features_name = feature_selection_main(
        train_data=train_data, configuration_file=configuration_file, folder_name=folder_name
    )
    return features_name


# -------------------------------- #
# Thanks to click commands not being able to be callable via command line and as a function at the same time (see https://github.com/pallets/click/issues/330), this function is doubled ..
def feature_selection_main(train_data, configuration_file, folder_name, with_train=False):
    """Run feature selection methods."""
    features_name = []
    init_logger(folder_name)
    log.info("Started feature selection")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    log.info("****************************** Load YAML ****************************** ")
    fs_type, fs_params = load_fs(general_configuration)
    log.info("****************************** Load fs ****************************** ")

    features_selection_path = MODELS_DIRECTORY / folder_name / FEATURES_SELECTION_DIRACTORY
    features_selection_path.mkdir(exist_ok=True, parents=True)
    log.info(f"Features list will be saved under {features_selection_path} ")
    log.info(f"{len(train_data.features)} features will be used  in feature selection  ")
    fs_configuration = general_configuration["FS"]
    fs_methods = {}
    for fs_type, fs_param in zip(fs_type, fs_params):
        log.info(f"{fs_type} :")
        configuration = copy.deepcopy(general_configuration)
        _fs_func(
            fs_type=fs_type,
            fs_param=fs_param,
            train_data=train_data,
            configuration=configuration,
            folder_name=folder_name,
            with_train=with_train,
            features_selection_path=features_selection_path,
        )
        features_name.append(fs_type)
        fs_methods[fs_type] = fs_param
    fs_configuration["FS_methods"] = fs_methods
    save_yml(fs_configuration, MODELS_DIRECTORY / folder_name / "FS_configuration.yml")
    return features_name


# -------------------------------- #
def _fs_func(  # noqa: CCR001
    fs_type,
    fs_param,
    train_data,
    configuration,
    folder_name,
    with_train,
    features_selection_path,
):
    """Apply features selection."""
    fs_class = get_model_by_name(fese, fs_type)
    fs_class(
        features=train_data.features,
        force_features=configuration["FS"]["force_features"],
        label_name=train_data.label,
        other_params=fs_param,
        folder_name=folder_name,
        n_feat=configuration["FS"]["n_feat"],
        fs_type=fs_type,
        with_train=with_train,
        features_selection_path=features_selection_path,
    ).select_features(train_data("features"), train_data("label"))


def _check_model_folder(folder_name):
    """Check if the checkpoint folder  exists or not."""
    model_folder_path = MODELS_DIRECTORY / folder_name
    if model_folder_path.exists():
        click.confirm(
            (
                f"The model folder with the name {folder_name}"
                "already exists. Do you want to continue the training"
                "but all the checkpoints will be deleted?"
            ),
            abort=True,
        )
        shutil.rmtree(model_folder_path)

    model_folder_path.mkdir(exist_ok=True, parents=True)
