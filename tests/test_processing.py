# type: ignore

"""This module includes the tests for the processing command."""
import os
from pathlib import Path
from typing import List, Optional, Tuple
from unittest import mock

from ig.cli.processing import process_main_data, process_other_data


@mock.patch("click.confirm")
def test_process_main_data(
    mock_click: mock.MagicMock,
    raw_data_paths: str,
    processed_dummy_data_path: Path,
    main_raw_data_paths: Optional[List[str]],
    processing_config_path: str,
    main_data_name: Optional[Tuple[str]] = ("dummy_data",),
) -> None:
    """This function tests the behavior of process_main_data function."""
    mock_click.return_value = "y"

    process_main_data(raw_data_paths, main_raw_data_paths, main_data_name, processing_config_path)

    assert processed_dummy_data_path.exists(), "Check process_main_data: No processed dataset !"
    assert set(os.listdir(processed_dummy_data_path)) == {
        "configuration.yml",
        "data_proc.log",
        "dummy_data.csv",
        "features.yml",
        "features_tracker.yml",
        "splits.csv",
        "train.csv",
    }, "Check process_main_data: Missing or inconvenient files !"


def test_process_other_data(
    test_data_path: str,
    processed_dummy_data_path: Path,
    other_data_name: Tuple[str] = ("dummy_dataset",),
    ignore_missing_features: bool = True,
) -> None:
    """This function tests the behavior of process_other_data function."""
    process_other_data(
        (test_data_path,), other_data_name, processed_dummy_data_path, ignore_missing_features
    )
    assert set(os.listdir(processed_dummy_data_path)) == {
        "configuration.yml",
        "data_proc.log",
        "dummy_data.csv",
        "dummy_dataset.csv",
        "features.yml",
        "features_tracker.yml",
        "splits.csv",
        "train.csv",
    }, "Check process_other_data: Missing or inconvenient files !"
