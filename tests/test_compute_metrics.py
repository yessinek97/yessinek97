# type: ignore
"""This module includes the test for compute_metrics.py module."""
import logging
from pathlib import Path
from typing import Iterable, Union

import click
import pytest
from click.testing import CliRunner

from ig.cli.compute_metrics import compute_metrics
from ig.utils.io import load_yml


def test_compute_metrics(
    caplog: pytest.LogCaptureFixture,
    test_data_path: str,
    models_path: Path,
    folder_name: str = "training_fixtures",
) -> None:
    """This function serves to test compute metrics function."""
    caplog.set_level(logging.INFO)

    runner: click.testing.CliRunner = CliRunner()
    params: Union[str, Iterable[str], None] = [
        "--test_data_paths",
        test_data_path,
        "--folder_name",
        folder_name,
    ]
    _ = runner.invoke(compute_metrics, params)
    experiment_path = (models_path) / folder_name
    best_exp_path = experiment_path / "best_experiment"
    generated_best_exp_data_file_path = best_exp_path / "eval" / "dummy_test_data.csv"
    generated_metrics_best_exp_path = best_exp_path / "eval" / "test_dummy_test_data_metrics.yaml"
    generated_kfold_exp_data_file_path = (
        experiment_path / "KfoldExperiment/train_features/XgboostModel/eval" / "dummy_test_data.csv"
    )
    generated_metrics_kfold_exp_path = (
        experiment_path
        / "KfoldExperiment/train_features/XgboostModel/eval"
        / "test_dummy_test_data_metrics.yaml"
    )
    output_log_file_path = experiment_path / "InfoEval.log"
    log_file = open(output_log_file_path).readlines()
    generated_metrics_best_exp = load_yml(generated_metrics_best_exp_path)
    generated_metrics_kfold_exp = load_yml(generated_metrics_kfold_exp_path)
    assert output_log_file_path.exists(), "Check compute-metrics generated log file !"
    assert log_file != [], "Check compute-metrics generated log file values !"
    assert (
        generated_best_exp_data_file_path.exists()
    ), "Check compute-metrics best experiment generated data file !"
    assert (
        generated_metrics_best_exp_path.exists()
    ), "Check compute-metrics best experiment generated metrics file !"
    assert set(generated_metrics_best_exp) == set(
        {
            "prediction_3": {
                "global": {
                    "topk": 0.5140562248995983,
                    "roc": 0.5065121041936671,
                    "logloss": 0.6931280847787857,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_1": {
                "global": {
                    "topk": 0.5020080321285141,
                    "roc": 0.5121361941791068,
                    "logloss": 0.6930929125547409,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_0": {
                "global": {
                    "topk": 0.4939759036144578,
                    "roc": 0.5012080193283093,
                    "logloss": 0.6931543987989426,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_4": {
                "global": {
                    "topk": 0.5140562248995983,
                    "roc": 0.5008640138242212,
                    "logloss": 0.6931546127796173,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_2": {
                "global": {
                    "topk": 0.5220883534136547,
                    "roc": 0.532816525064401,
                    "logloss": 0.6929997115135192,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_mean": {
                "global": {
                    "topk": 0.5220883534136547,
                    "roc": 0.5194083105329684,
                    "logloss": 0.6931042507886886,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_max": {
                "global": {
                    "topk": 0.5220883534136547,
                    "roc": 0.5366965871453943,
                    "logloss": 0.6930785865783692,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_min": {
                "global": {
                    "topk": 0.5180722891566265,
                    "roc": 0.5353685658970544,
                    "logloss": 0.6930262145996093,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_median": {
                "global": {
                    "topk": 0.5261044176706827,
                    "roc": 0.5104081665306646,
                    "logloss": 0.6931283011436462,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
        }
    ), "Check compute-metrics best experiment generated metrics values !"
    assert (
        generated_kfold_exp_data_file_path.exists()
    ), "Check compute-metrics kfold experiment generated data file !"
    assert (
        generated_metrics_kfold_exp_path.exists()
    ), "Check compute-metrics kfold experiment generated metrics file !"
    assert set(generated_metrics_kfold_exp) == set(
        {
            "prediction_3": {
                "global": {
                    "topk": 0.5180722891566265,
                    "roc": 0.5295124721995552,
                    "logloss": 0.6930839443206787,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_1": {
                "global": {
                    "topk": 0.5020080321285141,
                    "roc": 0.5062641002256036,
                    "logloss": 0.6931393909454345,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_0": {
                "global": {
                    "topk": 0.5020080321285141,
                    "roc": 0.5074081185298964,
                    "logloss": 0.6931199176311493,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_4": {
                "global": {
                    "topk": 0.5100401606425703,
                    "roc": 0.5118161890590249,
                    "logloss": 0.6931126534938812,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_2": {
                "global": {
                    "topk": 0.5301204819277109,
                    "roc": 0.5269124305988896,
                    "logloss": 0.6930885877609253,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_mean": {
                "global": {
                    "topk": 0.5100401606425703,
                    "roc": 0.5237283796540744,
                    "logloss": 0.6931082243919373,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_max": {
                "global": {
                    "topk": 0.5020080321285141,
                    "roc": 0.5137602201635226,
                    "logloss": 0.693111634850502,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_min": {
                "global": {
                    "topk": 0.5100401606425703,
                    "roc": 0.49980799692795086,
                    "logloss": 0.6931496757268906,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
            "prediction_median": {
                "global": {
                    "topk": 0.5220883534136547,
                    "roc": 0.5369925918814701,
                    "logloss": 0.693096471786499,
                    "precision": 0.498,
                    "recall": 1.0,
                    "f1": 0.664886515353805,
                }
            },
        }
    ), "Check compute-metrics kfold experiment generated metrics values !"
