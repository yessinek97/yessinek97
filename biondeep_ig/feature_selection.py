"""Module used to define the training command."""
import copy
import os
import shutil
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd

import biondeep_ig.src.feature_selection as fese
from biondeep_ig.src import CONFIGURATION_DIRECTORY
from biondeep_ig.src import DATAPROC_DIRACTORY
from biondeep_ig.src import FS_CONFIGURATION_DIRECTORY
from biondeep_ig.src import MODELS_DIRECTORY
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.logger import NeptuneLogs
from biondeep_ig.src.processing import Dataset
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
@click.option(
    "--test_data_path",
    "-test",
    type=str,
    required=True,
    help="Path to the dataset used in  evaluation.",
)
@click.option(
    "--unlabeled_path",
    "-unlabeled",
    type=str,
    required=False,
    help="Path to the unlabeled dataset.",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
def featureselection(
    train_data_path, test_data_path, unlabeled_path, configuration_file, folder_name
):
    """Feature selection process."""
    _check_model_folder(folder_name)
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["feature selection"], training_path=train_data_path, test_path=test_data_path
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)
    feature_selection_main(
        train_data_path, test_data_path, unlabeled_path, configuration_file, folder_name
    )


# -------------------------------- #
# Thanks to click commands not being able to be callable via command line and as a function at the same time (see https://github.com/pallets/click/issues/330), this function is doubled ..
def feature_selection_main(
    train_data_path, test_data_path, unlabeled_path, configuration_file, folder_name
):
    """Run feature selection methods."""
    init_logger(folder_name)
    log.info("Started feature selection")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    log.info("****************************** Load YAML ****************************** ")
    fs_type, fs_params = load_fs(general_configuration)
    log.info("****************************** Load fs ****************************** ")

    # displays = "####### fs Summary #######"
    save_yml(general_configuration, MODELS_DIRECTORY / folder_name / "configuration.yml")

    for fs_type, fs_param in zip(fs_type, fs_params):
        log.info(f"{fs_type} :")
        configuration = copy.deepcopy(general_configuration)
        # displays += f"\n fs :{experiment_name}: "
        _fs_func(
            fs_type=fs_type,
            fs_param=fs_param,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            unlabeled_path=unlabeled_path,
            configuration=configuration,
            folder_name=folder_name,
            log=log,
        )


# -------------------------------- #
def _fs_func(  # noqa: CCR001
    fs_type,
    fs_param,
    train_data_path,
    test_data_path,
    unlabeled_path,
    configuration,
    folder_name,
    log,
):

    train_columns = set(get_column_names(train_data_path))
    test_columns = set(get_column_names(test_data_path))
    features = train_columns & test_columns
    df_t = read(train_data_path, usecols=features)
    df_t.columns = df_t.columns.str.lower()

    # Get all useable columns based on min/max unique values:
    cols_to_remove = []
    iterator = df_t.columns.to_list()
    iterator.remove(configuration["label"].lower())
    for col in iterator:
        if (len(df_t[col].dropna().unique()) <= configuration["FS"]["min_unique_values"]) & (
            df_t[col].dtypes.type == np.object_
        ):
            cols_to_remove.append(col)
    for col in iterator:
        if (len(df_t[col].dropna().unique()) >= configuration["FS"]["max_unique_values"]) & (
            df_t[col].dtypes.type == np.object_
        ):
            cols_to_remove.append(col)

    # We have to temporarily write the file down to disk with the current structure ...
    with tempfile.NamedTemporaryFile(prefix="IMtmp_", suffix=".temp") as tmp_file:
        df_t.to_csv(tmp_file.name, sep="\t", index=False)

        # Usual processing of remaining features
        df_processed = (
            Dataset(
                data_path=FS_CONFIGURATION_DIRECTORY / str(tmp_file.name),
                features=[x.lower() for x in features],
                target=configuration["label"],
                configuration=configuration["processing"],
                is_train=True,
                experiment_path=MODELS_DIRECTORY / "temp",
            )
            .process_data()
            .data
        )

    label_s = df_processed[configuration["label"].lower()]
    df_processed = df_processed.drop(columns=cols_to_remove)
    df_processed = df_processed.drop(columns=[configuration["label"].lower()])

    df_processed = df_processed.loc[
        :, df_processed.isnull().mean() < configuration["FS"]["min_nonNaNValues"]
    ]

    forceexcludelist = load_yml(FS_CONFIGURATION_DIRECTORY / "FeatureExclude.yml")[
        "Feature_Exclude_List"
    ]
    forceexcludelist = [x.lower() for x in forceexcludelist]

    df_processed = df_processed[
        [x.lower() for x in df_processed.columns.to_list() if x.lower() not in forceexcludelist]
    ]
    features_a = df_processed.columns.to_list()

    fs_class = get_model_by_name(fese, fs_type)

    # features_list_paths = []
    # display += f"\n -{model_type} : \n"
    fs_class(
        features=features_a,
        force_features=configuration["FS"]["force_features"],
        label_name=configuration["label"],
        other_params=fs_param,
        folder_name=folder_name,
        n_feat=configuration["FS"]["n_feat"],
        fs_type=fs_type,
    ).select_features(df_processed, label_s)


def _check_model_folder(folder_name):
    model_folder_path = MODELS_DIRECTORY / folder_name
    if model_folder_path.exists():
        click.confirm(
            f"The model folder with the name {folder_name} already exists. Do you want to continue the training but all the checkpoints will be deleted?",
            abort=True,
        )
        shutil.rmtree(model_folder_path)

    model_folder_path.mkdir(exist_ok=True, parents=True)


def remove_processed_data(folder_name):
    """Remove data proc folder."""
    path = MODELS_DIRECTORY / folder_name / DATAPROC_DIRACTORY
    shutil.rmtree(path)


def log_summary_results(display):
    """Print summary results."""
    for line in display.split("\n"):
        log.info(line)


def get_safely_model_selection(general_configuration):
    """Load model selection column and check if it's included in the benchmark columns list or not."""
    model_selection = general_configuration.get("model_selection", "prediction_average")
    benchmark_column = general_configuration.get("benchmark_column", ["prediction_average"])
    if model_selection in benchmark_column:
        return model_selection
    raise KeyError(f"{model_selection}: is not available in the benchmark_column")


def read(file_path: Path, **kwargs):
    """Read data."""
    _, extension = os.path.splitext(file_path)
    if extension == ".csv":
        df = pd.read_csv(file_path, **kwargs)

    elif extension == ".tsv":
        df = pd.read_csv(file_path, sep="\t", **kwargs)

    elif extension == ".xlsx":
        df = pd.read_excel(file_path, **kwargs)

    else:
        raise ValueError(f"extension {extension} not supported")

    return df


def get_column_names(file_path: Path):
    """Get the columns names."""
    return read(file_path, nrows=1).columns.to_list()
