"""Module used to define helper functions for files and folders management (input/ouput)."""
import copy
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Union

import click
import pandas as pd
import yaml


def save_yml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save a dictionary to a yaml format."""
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a yaml file to a Python dictionary."""
    with open(file_path) as f:
        data = yaml.full_load(f)

    return data


def save_as_pkl(obj: Any, file_path: Union[str, Path]) -> None:
    """Save a python object as a pickle."""
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file_path: Union[str, Path]) -> Any:
    """Load a pickle previously saved by save_as_pkl function."""
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    return obj


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a json previously saved by save_as_json function."""
    with open(file_path) as json_file:
        return json.load(json_file)


def save_as_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save a dictionary as json file.

    Args:
        data: dictionary to save
        file_path: path to save the file
        save_kwargs: kwargs to forward to json.dump method.
            By default 'indent' is set to 4 for a better display.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


def read_data(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """This function is used to read the input dataset.

    Args:
        file_path: Path of the input dataset.

    Returns:
        df: The input dataframe.
    """
    logging.info("Loading: %s", file_path)

    extension = Path(file_path).suffix
    if extension == ".csv":
        df = pd.read_csv(file_path, **kwargs)

    elif extension == ".tsv":
        df = pd.read_csv(file_path, sep="\t", **kwargs)

    elif extension == ".xlsx":
        df = pd.read_excel(file_path, **kwargs)

    else:
        raise ValueError(f"extension {extension} not supported")

    return df


def check_and_create_folder(experiment_path: Path) -> None:
    """Check if the checkpoint folder  exists or not."""
    if experiment_path.exists():
        click.confirm(
            (
                f"The  folder with the name {experiment_path.name} already exists."
                "Do you want to continue but"
                "all the files will be deleted?"
            ),
            abort=True,
        )
        shutil.rmtree(experiment_path)

    experiment_path.mkdir(exist_ok=True, parents=True)


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


def check_model_folder(experiment_path: Path) -> None:
    """Check if the checkpoint folder  exists or not."""
    if experiment_path.exists():
        click.confirm(
            (
                f"The model folder with the name {experiment_path.name} already exists. "
                "Do you want to continue the training but "
                "all the checkpoints will be deleted?"
            ),
            abort=True,
        )
        shutil.rmtree(experiment_path)

    experiment_path.mkdir(exist_ok=True, parents=True)
