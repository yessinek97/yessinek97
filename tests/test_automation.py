"""This module includes the tests for all major commands in the IG framework."""
import logging
from pathlib import Path
from typing import Iterable, Union
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from ig.cli.compute_metrics import compute_metrics
from ig.cli.feature_selection import featureselection
from ig.cli.inference import inference
from ig.cli.multi_trainer import multi_train
from ig.cli.trainer import train, train_seed_fold, tune


@mock.patch("click.confirm")
def test_featureselection(
    mock_click: mock.MagicMock,
    train_data_path: str,
    config_path: str,
    folder_name: str,
) -> None:
    """This function runs feature selection to test errors."""
    mock_click.return_value = "n"

    runner = CliRunner()
    params = [
        "--train_data_path",
        train_data_path,
        "--folder_name",
        folder_name,
        "--configuration_file",
        config_path,
    ]
    featureselection_command = runner.invoke(featureselection, params)
    assert featureselection_command.exit_code == 0, "Check the feature selection command!"


@mock.patch("click.confirm")
def test_train(
    mock_click: mock.MagicMock,
    train_data_path: str,
    test_data_path: str,
    config_path: str,
    folder_name: str = "training_fixtures",
) -> None:
    """This function runs train command to test errors."""
    mock_click.return_value = "y"

    runner = CliRunner()
    params = [
        "--train_data_path",
        train_data_path,
        "--test_data_path",
        test_data_path,
        "--configuration_file",
        config_path,
        "--folder_name",
        folder_name,
    ]
    trainer_command = runner.invoke(train, params)
    assert trainer_command.exit_code == 0, "Check the train command!"


@mock.patch("click.confirm")
def test_train_seed_fold(
    mock_click: mock.MagicMock,
    train_data_path: str,
    test_data_path: str,
    train_seed_fold_config_path: str,
    folder_name: str,
) -> None:
    """This function runs train seed fold command to test errors."""
    mock_click.return_value = "y"
    runner = CliRunner()
    params = [
        "--train_data_path",
        train_data_path,
        "--test_data_path",
        test_data_path,
        "--configuration_file",
        train_seed_fold_config_path,
        "--folder_name",
        folder_name,
    ]
    train_seedfold_command = runner.invoke(train_seed_fold, params)
    assert train_seedfold_command.exit_code == 0, "Check the train seed fold command!"


@mock.patch("click.confirm")
def test_tune(
    mock_click: mock.MagicMock,
    train_data_path: str,
    test_data_path: str,
    tune_config_path: str,
    folder_name: str = "test_tune",
) -> None:
    """This function runs tune to test errors."""
    mock_click.return_value = "y"

    runner = CliRunner()
    params = [
        "--train_data_path",
        train_data_path,
        "--test_data_path",
        test_data_path,
        "--configuration_file",
        tune_config_path,
        "--folder_name",
        folder_name,
    ]
    tune_command = runner.invoke(tune, params)
    assert tune_command.exit_code == 0, "Check the tune command!"


def test_inference(
    test_data_path: str,
    folder_name: str = "training_fixtures",
    id_name: str = "id",
) -> None:
    """This function runs inference command to test errors."""
    runner = CliRunner()
    params = [
        "--test_data_path",
        test_data_path,
        "--folder_name",
        folder_name,
        "--id_name",
        id_name,
    ]
    inference_command = runner.invoke(inference, params)
    assert inference_command.exit_code == 0, "Check the Inference command!"


@mock.patch("click.confirm")
def test_compute_metrics_commands(
    caplog: pytest.LogCaptureFixture,
    test_data_path: str,
    folder_name: str = "training_fixtures",
) -> None:
    """This function runs compute metrics to test errors."""
    caplog.set_level(logging.INFO)

    runner: click.testing.CliRunner = CliRunner()
    params: Union[str, Iterable[str], None] = [
        "--test_data_paths",
        test_data_path,
        "--folder_name",
        folder_name,
    ]
    compute_metrics_command = runner.invoke(compute_metrics, params)
    assert compute_metrics_command.exit_code == 0, "Check the compute metrics command!"


@mock.patch("click.confirm")
def test_multi_train(
    mock_click: mock.MagicMock,
    train_data_path: str,
    test_data_path: str,
    multi_train_config_path: Path,
    default_config_path: Path,
    folder_name: str = "train_folder",
) -> None:
    """This function runs multi train command to test errors."""
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
    multi_train_command = runner.invoke(multi_train, params)
    assert multi_train_command.exit_code == 0, "Check the multi train command!"
