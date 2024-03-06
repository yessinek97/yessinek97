# type: ignore
"""This module includes the tests for feature selection pipeline."""
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from click.testing import CliRunner

from ig.dataset.dataset import Dataset
from ig.feature_selection import (
    _check_model_folder,
    _fs_func,
    feature_selection_main,
    featureselection,
)


@mock.patch("click.confirm")
def test_check_model_folder(
    mock_click: mock.MagicMock, models_path: Path, folder_name: str = "dummy_experiment"
) -> None:
    """This function tests the _check_model_folder function."""
    mock_click.return_value = "n"
    _check_model_folder(folder_name)

    model_folder_path = models_path / folder_name
    assert model_folder_path.exists()


@mock.patch("click.confirm")
def test_featureselection(
    mock_click: mock.MagicMock,
    models_path: Path,
    train_data_path: str,
    config_path: str,
    folder_name: str,
) -> None:
    """This function tests the featureselection function."""
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
    _ = runner.invoke(featureselection, params)
    assert (
        models_path / folder_name / "features/Fspca.txt"
    ).exists(), "Feature selection file not found!"


def test_fs_func(
    fs_methods: List[str],
    fs_params: List[Dict[str, Any]],
    train_dataset: Dataset,
    config: Dict[str, Any],
    folder_name: str,
    models_path: Path,
) -> None:
    """This function tests the _fs_func."""
    # Use PCA as a feature selection methods
    _fs_func(
        fs_type=fs_methods[0],
        fs_param=fs_params[0],
        train_data=train_dataset,
        configuration=config,
        folder_name=folder_name,
        features_selection_path=models_path / folder_name / "features",
        with_train=False,
    )
    features_list = [
        line.strip()
        for line in (models_path / folder_name / "features/Fspca.txt").open().read().splitlines()
    ]

    expected_features_list = [
        "mut_mswhim1",
        "tested_score_biondeep_mhci",
        "tested_presentation_biondeep_mhci",
        "refractivity_nontested_pep_global_whole",
        "wt_zscales3",
        "z5_nontested_pep_global_whole",
        "diff_fasgai_local_flex_pep_mhci",
        "pram900101_mut",
        "zimj680101_mut",
        "expression",
        "tested_best_rank_biondeep_flanks_mhcii",
        "nontested_hyd_pep_mhci",
        "charge_phys_nontested_pep_moment_whole",
        "nontested_zscales2_pep_mhci",
        "diff_fasgai_hydr",
        "mut_fasgai_local_flex",
        "neoantigenquality",
        "fasgai1_tc3_mhci",
        "charge_phys_nontested_pep_moment_mhci",
        "fasgai2_tc3_mhci",
        "nontested_mswhim2_pep_mhci",
        "diff_fasgai_alpha_turn_pep_mhci",
    ]
    assert (
        models_path / folder_name / "features/Fspca.txt"
    ).exists(), "Feature selection file not found!"
    assert set(expected_features_list).difference(features_list) == set()


def test_feature_selection_main(
    train_dataset: Dataset,
    config: Dict[str, Any],
    models_path: Path,
    folder_name: str,
    with_train: bool = False,
) -> None:
    """This function tests the feature_selection_main function."""
    feature_names = feature_selection_main(train_dataset, config, folder_name, with_train)
    output_config_path = models_path / folder_name / "FS_configuration.yml"

    assert feature_names == ["Fspca"]

    assert output_config_path.exists()
