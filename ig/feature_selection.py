"""Module used to define the training command."""
import copy
import shutil
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Union

import click

import ig.src.feature_selection as fese
from ig import CONFIGURATION_DIRECTORY, FEATURES_SELECTION_DIRECTORY, MODELS_DIRECTORY
from ig.dataset.dataset import Dataset
from ig.src.logger import NeptuneLogs, get_logger, init_logger
from ig.src.utils import get_model_by_name, load_fs, load_yml, save_yml

log: Logger = get_logger("FeatureSelection")


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
@click.pass_context
def featureselection(
    ctx: Union[click.core.Context, Any],
    train_data_path: str,
    configuration_file: str,
    folder_name: str,
) -> None:
    """Feature selection process."""
    _check_model_folder(folder_name)
    configuration_file_path = CONFIGURATION_DIRECTORY / configuration_file
    general_configuration = load_yml(configuration_file_path)
    experiment_path = MODELS_DIRECTORY / folder_name
    init_logger(experiment_path, "FeatureSelection")

    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["feature selection"], training_path=train_data_path, test_path=None
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)

    train_data = Dataset(
        click_ctx=ctx,
        data_path=train_data_path,
        configuration=general_configuration,
        is_train=True,
        experiment_path=experiment_path,
        force_gcp=False,
    ).load_data()
    feature_selection_main(
        train_data=train_data, general_configuration=general_configuration, folder_name=folder_name
    )


# -------------------------------- #
# Thanks to click commands not being able to be callable via command line and as a function at the same time (see https://github.com/pallets/click/issues/330), this function is doubled ..
def feature_selection_main(
    train_data: Dataset,
    general_configuration: Dict[str, Any],
    folder_name: str,
    with_train: bool = False,
) -> List[str]:
    """Run feature selection methods."""
    features_name: List[str] = []

    log.info("Started feature selection")
    log.info("****************************** Load YAML ****************************** ")
    fs_type, fs_params = load_fs(general_configuration)
    log.info("****************************** Load fs ****************************** ")

    features_selection_path = MODELS_DIRECTORY / folder_name / FEATURES_SELECTION_DIRECTORY
    features_selection_path.mkdir(exist_ok=True, parents=True)
    log.info("Features list will be saved under %s ", features_selection_path)
    log.info("%s features will be used  in feature selection  ", len(train_data.features))
    fs_configuration = general_configuration["FS"]
    fs_methods: Dict[str, Dict[str, Any]] = {}
    for fs_types, fs_param in zip(fs_type, fs_params):
        log.info("%s :", fs_types)
        configuration = copy.deepcopy(general_configuration)
        _fs_func(
            fs_type=fs_types,
            fs_param=fs_param,
            train_data=train_data,
            configuration=configuration,
            folder_name=folder_name,
            with_train=with_train,
            features_selection_path=features_selection_path,
        )
        features_name.append(fs_types)
        fs_methods[fs_types] = fs_param
    fs_configuration["FS_methods"] = fs_methods
    save_yml(fs_configuration, MODELS_DIRECTORY / folder_name / "FS_configuration.yml")
    return features_name


# -------------------------------- #
def _fs_func(  # noqa: CCR001
    fs_type: str,
    fs_param: Dict[str, Any],
    train_data: Dataset,
    configuration: Dict[str, Any],
    folder_name: str,
    with_train: bool,
    features_selection_path: Path,
) -> None:
    """Apply features selection."""
    fs_class = get_model_by_name(fese, fs_type)
    fs_class(
        features=train_data.features,
        force_features=configuration["FS"].get("force_features", []),
        label_name=train_data.label,
        other_params=fs_param,
        folder_name=folder_name,
        n_feat=configuration["FS"]["n_feat"],
        fs_type=fs_type,
        with_train=with_train,
        features_selection_path=features_selection_path,
        keep_same_features_number=configuration["FS"].get("keep_features_number", False),
        separate_forced_features=configuration["FS"].get("separate_forced_features", False),
        n_thread=configuration["FS"].get("n_thread", -1),
    ).select_features(train_data("features"), train_data("label"))


def _check_model_folder(folder_name: str) -> None:
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
