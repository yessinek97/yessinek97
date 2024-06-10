# type: ignore
"""This module includes the tests for test_multi_trainer.py."""
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
from unittest import mock

import click
from click.testing import CliRunner

from ig.cli.multi_trainer import experiments_evaluation, multi_train
from ig.utils.io import generate_experiment_configuration


@click.pass_context
def multi_train_context() -> str:
    """Dummy context."""
    return "dummy context"


def test_generate_experiment_configuration(
    default_config: Dict[str, Any],
    multi_train_config: Dict[str, Any],
    test_experiment_path: Path,
    experiment_name: str = "experiment1",
) -> None:
    """This function tests the behavior of generate_experiment_configuration function."""
    experiment_configuration_path = test_experiment_path / "configuration"
    # Make empty configuration folder
    os.makedirs(experiment_configuration_path, exist_ok=True)
    path = generate_experiment_configuration(
        default_config,
        multi_train_config["experiments"][experiment_name],
        experiment_configuration_path,
        experiment_name,
    )
    assert (
        path == "tests/models/dummy_experiment/configuration/experiment1.yml"
    ), "Check generate _experiment_configuration!"


def test_experiments_evaluation(
    experiment_paths: List[Path],
    multi_train_config: Dict[str, Any],
    multi_train_experiment_path: Path,
) -> None:
    """This function tests the behavior of experiments_evaluation."""
    params = multi_train_config["params"]
    output_metrics_dir_path = multi_train_experiment_path / "results"
    experiments_evaluation(experiment_paths, multi_train_experiment_path, params)

    assert set(os.listdir(output_metrics_dir_path)) == {
        "f1.csv",
        "logloss.csv",
        "precision.csv",
        "recall.csv",
        "roc.csv",
        "topk.csv",
        "topk_20_patientid.csv",
        "topk_patientid.csv",
        "topk_retrieval_20_patientid.csv",
        "topk_retrieval_patientid.csv",
    }, "Check experiments_evaluation!"


@mock.patch("click.confirm")
def test_multi_train(
    mock_click: mock.MagicMock,
    train_data_path: str,
    test_data_path: str,
    multi_train_config_path: Path,
    default_config_path: Path,
    models_path: Path,
    folder_name: str = "train_folder",
) -> None:
    """This function tests the behavior of multi-train command."""
    mock_click.return_value = "y"
    multi_train_config = multi_train_config_path.name
    default_config = default_config_path.name
    runner = CliRunner()
    params: Union[str, Iterable[str], None] = [
        "--train_data_path",
        train_data_path,
        "--test_data_path",
        test_data_path,
        "--configuration_file",
        multi_train_config,
        "--default_configuration_file",
        default_config,
        "--folder_name",
        folder_name,
    ]
    _ = runner.invoke(multi_train, params)

    assert set(os.listdir(models_path / folder_name)) == {
        "configuration",
        "experiment1",
        "experiment2",
        # "multi-train-configuration.yml",
        "multi-train.log",
        "results",
    }, "Check the multi-train command!"
