# type: ignore
"""This module includes the tests for trainer module."""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Union
from unittest import mock

import click
import pandas as pd
import pytest
from click.testing import CliRunner

from ig.cli.trainer import (
    _check_model_folder,
    _generate_single_exp_config,
    _save_best_experiment,
    _train_func,
    compute_comparison_score,
    eval_comparison_score,
    load_datasets,
    parse_comparison_score_metrics_to_df,
    remove_unnecessary_folders,
    train,
    tune,
)
from ig.dataset.dataset import Dataset


class Dummycontext(click.Option):
    """A dummy custom class that delays the default lookup from context until its creation."""

    def __init__(self, default_name: str):
        """Initialize the context class."""
        self.default_name = default_name

    def get_default(self, ctx: Union[click.core.Context, Any]) -> click.core.Context:
        """Get the default context."""
        self.default = ctx.obj[self.default_name]
        return super().get_default(ctx)


def test_load_datasets(
    train_data_path: str,
    test_data_path: str,
    config: Dict[str, Any],
    test_experiment_path: Path,
) -> None:
    """This function tests the behavior of load_datasets function."""
    test_context = Dummycontext(default_name="dummy")
    train_data, test_data = load_datasets(
        click_ctx=test_context,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        general_configuration=config,
        experiment_path=test_experiment_path,
        force=True,
    )
    assert isinstance(train_data, Dataset), "There is a problem building the training dataset!"
    assert isinstance(test_data, Dataset), "There is a problem building the test dataset!"


def expected_comparison_score_output(
    comparison_score: Union[None, str]
) -> Union[None, Dict[str, Any]]:
    """Dummy expected_comparison_score_output."""
    if comparison_score is None:
        output = None
    else:

        output = {
            "experiments": {
                0: "KfoldExperiment",
                1: "KfoldExperiment",
                2: "KfoldExperiment",
                3: "KfoldExperiment",
                4: "KfoldExperiment",
                5: "KfoldExperiment",
                6: "KfoldExperiment",
                7: "KfoldExperiment",
                8: "KfoldExperiment",
                9: "KfoldExperiment",
                10: "test",
            },
            "prediction": {
                0: "prediction_0",
                1: "prediction_0",
                2: "prediction_1",
                3: "prediction_1",
                4: "prediction_2",
                5: "prediction_2",
                6: "prediction_3",
                7: "prediction_3",
                8: "prediction_4",
                9: "prediction_4",
                10: "prediction",
            },
            "split": {
                0: "train",
                1: "validation",
                2: "train",
                3: "validation",
                4: "train",
                5: "validation",
                6: "train",
                7: "validation",
                8: "train",
                9: "validation",
                10: "test",
            },
            "topk": {
                0: 0.5404040404040404,
                1: 0.5081967213114754,
                2: 0.5333333333333333,
                3: 0.5306122448979592,
                4: 0.5048076923076923,
                5: 0.6470588235294118,
                6: 0.5534883720930233,
                7: 0.4772727272727273,
                8: 0.5463414634146342,
                9: 0.5370370370370371,
                10: 0.5381526104417671,
            },
            "roc": {
                0: 0.539645156225467,
                1: 0.467896174863388,
                2: 0.5169482846902202,
                3: 0.5469387755102041,
                4: 0.5057597105864433,
                5: 0.6430367018602312,
                6: 0.5325942884684901,
                7: 0.5056818181818181,
                8: 0.5334788189987163,
                9: 0.5039941902687,
                10: 0.5360885774172387,
            },
            "type": {
                0: "tested_score_biondeep_mhci",
                1: "tested_score_biondeep_mhci",
                2: "tested_score_biondeep_mhci",
                3: "tested_score_biondeep_mhci",
                4: "tested_score_biondeep_mhci",
                5: "tested_score_biondeep_mhci",
                6: "tested_score_biondeep_mhci",
                7: "tested_score_biondeep_mhci",
                8: "tested_score_biondeep_mhci",
                9: "tested_score_biondeep_mhci",
                10: "tested_score_biondeep_mhci",
            },
        }
    return output


@pytest.mark.parametrize("comparison_score", [None, "tested_score_biondeep_mhci"])
def test_eval_comparison_score(
    config: Dict[str, Any],
    comparison_score: Union[None, str],
    train_data_path: str,
    test_data_path: str,
    test_experiment_path: Path,
    dataset_name: str = "dummy",
) -> None:
    """This function tests the behavior of eval_comparison_score function."""
    comparison_score = config["evaluation"]["comparison_score"] = comparison_score
    expected_output = expected_comparison_score_output(comparison_score)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    result: pd.DataFrame = eval_comparison_score(
        configuration=config,
        train_data=train_data,
        test_data=test_data,
        experiment_path=test_experiment_path,
        dataset_name=dataset_name,
    )

    if comparison_score is not None:
        assert result.to_dict() == expected_output, "Check comparison score values!"
        comparison_score_path = (
            test_experiment_path / "comparison_score" / f"eval_{comparison_score}.yml"
        )

        assert comparison_score_path.exists(), "Failed generating the Comparison score file!"
    else:
        assert result == expected_output, "Check comparison score values!"


def test__generate_single_exp_config(
    config: Dict[str, Any],
    experiment_name: str,
    experiment_params: Dict[str, Any],
    model_type: str,
    model_params: Dict[str, Any],
) -> None:
    """This function tests the behavior of _generate_single_exp_config function."""
    new_config_file = _generate_single_exp_config(
        config, experiment_name, experiment_params, model_type, model_params
    )

    assert "feature_paths" not in new_config_file.keys(), "Check the single experiment config file!"
    if isinstance(new_config_file["processing"], dict):

        assert (
            set((new_config_file["processing"]).keys()).intersection(["folds", "seeds"]) == set()
        ), "Check the single experiment config file!"
    assert new_config_file["experiments"] == {
        experiment_name: experiment_params
    }, "Check the experiment name and parameters!"
    assert new_config_file["model_type"] == model_type, "Check the model type configuration!"
    assert new_config_file["model"] == model_params, "Check the model parameters configuration!"


def test__train_func(
    experiment_name: str,
    experiment_params: Dict[str, Any],
    model_type: str,
    model_params: Dict[str, Any],
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: Dict[str, Any],
    logger: logging.Logger,
    training_display: str,
    folder_name: str = "training_fixtures",
) -> None:
    """This function tests the behavior of _train_func."""
    results, display = _train_func(
        experiment_name,
        experiment_params,
        model_type,
        model_params,
        train_dataset,
        test_dataset,
        "",
        config,
        folder_name,
        logger,
    )

    # Check that the ouput scores are as expected
    display_scores = re.findall(r"\d+\.\d{3}", display)
    training_display_scores = re.findall(r"\d+\.\d{3}", training_display)
    assert display_scores == training_display_scores, "train_func output scores has changed!"

    # Check the display message
    display = "".join(display.splitlines()).replace(" ", "")
    assert display == training_display, "Check train_func display message!"

    # Check the ouput results
    assert len(results) == 1, "train_func should output the results for 1 experiment only!"
    assert list(results[0].keys()) == [
        "validation",
        "test",
        "statistic",
    ], "Check train_func evaluation output format!"


def test_compute_comparison_score(
    caplog: pytest.LogCaptureFixture,
    input_data_path: str,
    label_name: str = "cd8_any",
    column_name: str = "presentation_score",
) -> None:
    """This function tests the behavior of compute_comparison_score function."""
    runner = CliRunner()
    params = [
        "--data_path",
        input_data_path,
        "--label_name",
        label_name,
        "--column_name",
        column_name,
    ]
    _ = runner.invoke(compute_comparison_score, params)
    expected_output = caplog.messages
    assert expected_output == [
        "Evaluating presentation_score in test_data.csv",
        "The new Topk",
        "Topk : 0.6923076923076923 where 9 positive label is captured ",
    ]


def test_parse_comparison_score_metrics_to_df(
    metrics: Dict[str, Any], comparison_score: str = "tested_score_biondeep_mhci"
) -> None:
    """This function tests the behavior of parse_comparison_score_metrics_to_df."""
    result = parse_comparison_score_metrics_to_df(metrics, comparison_score)
    expected_result = {
        "experiments": {0: "KfoldExperiment", 1: "KfoldExperiment", 2: "test"},
        "prediction": {0: "prediction_0", 1: "prediction_0", 2: "prediction"},
        "split": {0: "train", 1: "validation", 2: "test"},
        "topk": {0: 0.5797101449275363, 1: 0.48760330578512395, 2: 0.5381526104417671},
        "roc": {0: 0.5243918219461697, 1: 0.5204689602152605, 2: 0.5360885774172387},
        "type": {
            0: "tested_score_biondeep_mhci",
            1: "tested_score_biondeep_mhci",
            2: "tested_score_biondeep_mhci",
        },
    }
    assert (
        result.to_dict() == expected_result
    ), "Check parse_comarison_score_metrics_to_df function!"


@mock.patch("click.confirm")
def test__check_model_folder(mock_click: mock.MagicMock, test_experiment_path: Path) -> None:
    """This function tests the behavior of _check_model_folder."""
    mock_click.return_value = "n"
    _check_model_folder(test_experiment_path / "new")

    assert (test_experiment_path / "new").exists(), "Experiment folder doesn't exist!"


def test_remove_unnecessary_folders(test_experiment_path, data_proc_dir, remove=True):
    """This function tests the behavior of remove_unnecessary_folders."""
    kfold_dir_path = test_experiment_path / "testing" / "KfoldExperiment"
    dkfold_dir_path = test_experiment_path / "testing" / "DoubleKfold"
    kfoldmultiseed_dir_path = test_experiment_path / "testing" / "KfoldMultiSeedExperiment"
    single_model_dir_path = test_experiment_path / "testing" / "SingleModel"

    os.makedirs(kfoldmultiseed_dir_path, exist_ok=True)
    os.makedirs(kfold_dir_path, exist_ok=True)

    os.makedirs(dkfold_dir_path, exist_ok=True)
    os.makedirs(single_model_dir_path, exist_ok=True)
    os.makedirs(test_experiment_path / "testing" / "data_proc", exist_ok=True)
    test_data = pd.DataFrame()
    test_data.to_csv(test_experiment_path / "testing" / "data_proc" / "data_file.csv")
    remove_unnecessary_folders(data_proc_dir.parent, remove)
    assert not ((kfold_dir_path).exists()), "Check remove_unnecessary_folders function!"

    assert not ((dkfold_dir_path).exists()), "Check remove_unnecessary_folders function!"
    assert not ((kfoldmultiseed_dir_path).exists()), "Check remove_unnecessary_folders function!"
    assert (data_proc_dir).exists(), "Check remove_unnecessary_folders function!"
    assert (os.listdir(data_proc_dir)) == [], "Check remove_unnecessary_folders function!"


def test__save_best_experiment(best_experiment_id: List[str], training_fixtures_path: Path) -> None:
    """This function tests the behavior of _save_best_experiment function."""
    _save_best_experiment(best_experiment_id, training_fixtures_path)

    assert set(os.listdir(training_fixtures_path / "best_experiment")) == {
        "features.txt",
        "configuration.yml",
        "checkpoint",
        "eval",
        "prediction",
    }, "Check _save_best_experiment_function!"


@mock.patch("click.confirm")
def test_train(
    mock_click: mock.MagicMock,
    models_path: Path,
    train_data_path: str,
    test_data_path: str,
    config_path: str,
    folder_name: str = "training_fixtures",
) -> None:
    """This function tests the behavior of train."""
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
    _ = runner.invoke(train, params)
    experiment_path = models_path / folder_name
    experiment_directory = os.listdir(experiment_path)
    training_log_file_path = experiment_path / "InfoRun.log"
    training_log_file = open(training_log_file_path).readlines()
    assert training_log_file_path.exists(), "The training log file does not exist!"
    assert set(experiment_directory) == set(
        {
            "FS_configuration.yml",
            "InfoRun.log",
            "KfoldExperiment",
            "best_experiment",
            "comparison_score",
            "configuration.yml",
            "data_proc",
            "features",
            "models_configuration",
            "run_configuration.yml",
            "results.csv",
        }
    ), "Check training directory !"
    assert training_log_file != [], "Check the training log file values !"


@mock.patch("click.confirm")
def test_train_seed_fold(
    mock_click: mock.MagicMock,
    models_path: Path,
    train_data_path: str,
    test_data_path: str,
    train_seed_fold_config_path: str,
    folder_name: str,
) -> None:
    """This function tests the behavior of train_seed_fold."""
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
    _ = runner.invoke(train, params)
    experiment_path = models_path / folder_name
    experiment_directory = os.listdir(experiment_path)
    training_log_file_path = experiment_path / "InfoRun.log"
    training_log_file = open(training_log_file_path).readlines()
    assert training_log_file_path.exists(), "The training log file does not exist!"
    assert set(experiment_directory) == set(
        {
            "InfoRun.log",
            "KfoldExperiment",
            "best_experiment",
            "comparison_score",
            "configuration.yml",
            "data_proc",
            "features",
            "models_configuration",
            "run_configuration.yml",
            "results.csv",
        }
    ), "Check training directory for train-seed-fold !"
    assert training_log_file != [], "Check the training log file values !"


@mock.patch("click.confirm")
def test_tune(
    mock_click: mock.MagicMock,
    models_path: Path,
    train_data_path: str,
    test_data_path: str,
    tune_config_path: str,
    folder_name: str = "test_tune",
) -> None:
    """This function tests the behavior of tune."""
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
    _ = runner.invoke(tune, params)
    experiment_path = models_path / folder_name
    experiment_directory = os.listdir(experiment_path)
    tuning_log_file_path = experiment_path / "InfoRun.log"
    tuning_log_file = open(tuning_log_file_path).readlines()
    assert tuning_log_file_path.exists(), "The tuning log file does not exist!"
    assert set(experiment_directory) == set(
        {
            "InfoRun.log",
            "XgboostModel",
            "configuration.yml",
            "data_proc",
            "features",
            "models_configuration",
            "results.csv",
        }
    ), "Check tuning directory for tune command!"
    assert tuning_log_file != [], "Check the tune log file values !"

    assert (models_path / folder_name / "results.csv").exists(), "Check Tune command !"
