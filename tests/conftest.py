# type: ignore
# pylint: disable=redefined-outer-name
"""Fixtures and variable used to test different codes in the project."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import click
import pandas as pd
import pytest

from ig import CONFIGURATION_DIRECTORY, DATA_DIRECTORY
from ig.dataset.dataset import Dataset
from ig.utils.io import load_yml
from ig.utils.logger import get_logger


@pytest.fixture(scope="session")
def current_path() -> Path:
    """Get Current Path to access the toy dataset."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def helper_dummy_data_path() -> Path:
    """helper_dummy_data_path."""
    path = "tests/fixtures/helper_dummy_data.csv"
    return path


@pytest.fixture(scope="session")
def helper_dummy_df():
    """Helper dummy dataframe."""
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
            "col3": [1.0, 2.0, 3.0],
            "col4": [True, False, True],
            "col5": ["NA", "NaN", "NAN"],
        }
    )
    return data


@pytest.fixture(scope="session")
def file_paths() -> List[str]:
    """Dummy test file paths."""
    return [
        "tests/fixtures/test_data.csv",
        "tests/fixtures/test_data.tsv",
        "tests/fixtures/test_data.xlsx",
    ]


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    """Dummy test data."""
    return pd.read_csv("tests/fixtures/test_data.csv")


@pytest.fixture(scope="session")
def test_go_embeddings_df() -> pd.DataFrame:
    """Dummy test data with go embedded vectors."""
    return pd.read_csv("tests/fixtures/test_data_go_embeddings.csv")


@pytest.fixture(scope="session")
def sequence() -> str:
    """Dummy Gene Ontology sequence."""
    return "GO:0005575;GO:0005622;GO:0005623;GO:0005634;GO:0043226"


@pytest.fixture(scope="session")
def input_go_terms() -> str:
    """Dummy Input Gene Ontology column names space separated."""
    return "go_term_cc go_term_mf go_term_bp"


@pytest.fixture(scope="session")
def input_data_path() -> str:
    """Dummy Input dataset path."""
    return "tests/fixtures/test_data.csv"


@pytest.fixture(scope="session")
def techniques() -> List[str]:
    """Test dimensionality reduction techniques."""
    return ["pca", "lsa", "tsne"]


@pytest.fixture(scope="session")
def n_components() -> int:
    """Test number of components for dimensionality reduction."""
    return 3


@pytest.fixture(scope="session")
def embedding_features() -> List[str]:
    """Test embedding features."""
    return ["go_term_cc_embed_vector"]


@pytest.fixture(scope="session")
def fs_methods() -> List[str]:
    """Test feature selection methods."""
    return [
        "Fspca",
        "Fsxgboost",
        "Fsxgboostshap",
        "Fsrfgini",
        "Fscossim",
        "Fscorr",
        "Fssfmlr",
        "Fsrfelr",
        "Fssfmxgboost",
        "Fsrfexgboost",
        "Fsrelief",
        "Fsmi",
        "Fsboruta",
    ]


@pytest.fixture(scope="session")
def fs_params() -> List[Any]:
    """Test feature selection methods."""
    return [
        {"SamplesPerPCA": 1},
        {"max_depth": 7, "learning_rate": 0.03, "nthread": 24, "rand_seed": 1994},
        {"max_depth": 7, "learning_rate": 0.03, "nthread": 24, "rand_seed": 1994},
        {
            "max_depth": 7,
            "learning_rate": 0.03,
            "nthread": 24,
            "rand_seed": 1994,
            "n_estimators": 100,
            "max_features": 30,
            "criterion": "gini",
        },
        None,
        None,
        {"n_thread": 24},
        {"n_thread": 24},
        {"max_depth": 7, "learning_rate": 0.03, "nthread": 24, "rand_seed": 1994},
        {"max_depth": 7, "learning_rate": 0.03, "nthread": 24, "rand_seed": 1994},
        {"n_threads": 24, "rand_seed": 1994},
        None,
        {
            "max_depth": 7,
            "learning_rate": 0.03,
            "nthread": 24,
            "rand_seed": 1994,
            "n_estimators": "auto",
            "max_iter": 10,
            "max_features": 30,
            "criterion": "gini",
        },
    ]


@pytest.fixture(scope="session")
def config_path() -> str:
    """Test dummy config path."""
    path = "quickstart_train.yml"

    return path


@pytest.fixture(scope="session")
def train_seed_fold_config_path() -> str:
    """Test dummy train_seed_fold config path."""
    path = "train_seed_fold.yml"

    return path


@pytest.fixture(scope="session")
def tune_config_path() -> str:
    """Test dummy tune config path."""
    path = "tune_configuration.yml"

    return path


@pytest.fixture(scope="session")
def models_path() -> Path:
    """Test dummy model path."""
    path = Path("tests/models")

    return path


@pytest.fixture(scope="session")
def config() -> Dict[str, Any]:
    """Test dummy config."""
    general_configuration = load_yml("tests/configuration/quickstart_train.yml")

    return general_configuration


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    """Test dummy default config path."""
    path = Path("tests/configuration/default_configuration.yml")

    return path


@pytest.fixture(scope="session")
def default_config(default_config_path: Path) -> Dict[str, Any]:
    """Test dummy default config."""
    general_configuration = load_yml(default_config_path)

    return general_configuration


@pytest.fixture(scope="session")
def multi_train_experiment_path() -> Path:
    """Test dummy multi_train config."""
    path = Path("tests/models/train_folder")

    return path


@pytest.fixture(scope="session")
def experiment_paths(multi_train_experiment_path: Path) -> List[Path]:
    """Test dummy experiment paths."""
    paths = [
        multi_train_experiment_path / "experiment1",
        multi_train_experiment_path / "experiment2",
    ]

    return paths


@pytest.fixture(scope="session")
def multi_train_config_path() -> Path:
    """Test dummy multi_train config path."""
    path = Path("tests/configuration/multi_train_configuration.yml")

    return path


@pytest.fixture(scope="session")
def multi_train_config(multi_train_config_path: Path) -> Dict[str, Any]:
    """Test dummy multi_train config."""
    general_configuration = load_yml(multi_train_config_path)

    return general_configuration


@pytest.fixture(scope="session")
def test_experiment_path() -> Path:
    """Dummy train data."""
    path = Path("tests/models/dummy_experiment")
    return path


@pytest.fixture(scope="session")
def training_fixtures_path() -> Path:
    """Dummy train fixtures path."""
    path = Path("tests/models/training_fixtures")
    return path


@pytest.fixture(scope="session")
def data_proc_dir(test_experiment_path: Path) -> Path:
    """A dummy temporary directory used in tests."""
    path = test_experiment_path / "testing/data_proc"
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def train_data_path() -> str:
    """Dummy train_data_path path."""
    path = "tests/fixtures/dummy_train_data.csv"

    return path


@pytest.fixture(scope="session")
def test_data_path() -> str:
    """Dummy test_data_path path."""
    path = "tests/fixtures/dummy_test_data.csv"

    return path


@pytest.fixture(scope="session")
@click.pass_context
def _dummy_context() -> click.core.Context:
    """Dummy context."""


@pytest.fixture(scope="session")
def train_dataset(
    test_experiment_path: Path, config: Dict[str, Any], train_data_path: str
) -> Dataset:
    """Dummy train dataset."""
    train_data = Dataset(
        click_ctx=_dummy_context,
        data_path=train_data_path,
        configuration=config,
        is_train=True,
        experiment_path=test_experiment_path,
    ).load_data()
    return train_data


@pytest.fixture(scope="session")
def test_dataset(
    test_experiment_path: Path, config: Dict[str, Any], test_data_path: str
) -> Dataset:
    """Dummy test dataset."""
    test_data = Dataset(
        click_ctx=_dummy_context,
        data_path=test_data_path,
        configuration=config,
        is_train=False,
        experiment_path=test_experiment_path,
    ).load_data()
    return test_data


@pytest.fixture(scope="session")
def bucket_path() -> str:
    """Dummy bucket path."""
    path = "gs://biondeep-data/IG/tests/fixtures"
    return path


@pytest.fixture(scope="session")
def local_path() -> str:
    """Dummy local path."""
    return "tests/fixtures"


@pytest.fixture(scope="session")
def experiment_name() -> str:
    """Dummy experiment name."""
    return "KfoldExperiment"


@pytest.fixture(scope="session")
def folder_name() -> str:
    """Dummy folder name."""
    return "dummy_experiment"


@pytest.fixture(scope="session")
def experiment_params() -> Dict[str, Any]:
    """Dummy experiment parameters."""
    return {
        "split_column": "fold",
        "plot_shap_values": False,
        "plot_kfold_shap_values": True,
        "operation": ["mean", "max", "min", "median"],
        "statics": ["mean", "max", "min"],
    }


@pytest.fixture(scope="session")
def model_type() -> str:
    """Dummy model type."""
    return "XgboostModel"


@pytest.fixture(scope="session")
def training_display() -> str:
    """Expected training display."""
    return "-XgboostModel:*Features:train_featuresprediction_mean:Validation:0.475Test:0.506"


@pytest.fixture(scope="session")
def model_params() -> Dict[str, Any]:
    """Dummy model parameters."""
    return {
        "general_params": {
            "num_boost_round": 1000,
            "verbose_eval": 10,
            "early_stopping_rounds": 15,
        },
        "model_params": {
            "alpha": 0.01,
            "booster": "gbtree",
            "colsample_bytree": 1,
            "eta": 0.05,
            "eval_metric": "logloss",
            "gamma": 0.2,
            "lambda": 3,
            "max_depth": 5,
            "min_child_weight": 19,
            "nthread": 16,
            "objective": "binary:logistic",
            "seed": 2020,
            "subsample": 1,
        },
    }


@pytest.fixture(scope="session")
def logger() -> logging.Logger:
    """Dummy testing logger."""
    log = get_logger("Testing pipeline")
    return log


@pytest.fixture(scope="session")
def metrics() -> Dict[str, Any]:
    """Dummy testing metrics."""
    return {
        "KfoldExperiment": {
            "train_0": {
                "tested_score_biondeep_mhci": {
                    "global": {
                        "topk": 0.5797101449275363,
                        "roc": 0.5243918219461697,
                        "logloss": 16.440633475412127,
                        "precision": 0.5736434108527132,
                        "recall": 0.5362318840579711,
                        "f1": 0.5543071161048689,
                    }
                }
            },
            "validation_0": {
                "tested_score_biondeep_mhci": {
                    "global": {
                        "topk": 0.48760330578512395,
                        "roc": 0.5204689602152605,
                        "logloss": 16.71693728979197,
                        "precision": 0.5,
                        "recall": 0.4380165289256198,
                        "f1": 0.46696035242290745,
                    }
                }
            },
        },
        "test": {
            "test": {
                "tested_score_biondeep_mhci": {
                    "patientid": {
                        "Patient002": {
                            "topk": 0.6268656716417911,
                            "roc": 0.5790491984521835,
                            "logloss": 13.99390048020761,
                            "precision": 0.6491228070175439,
                            "recall": 0.5522388059701493,
                            "f1": 0.5967741935483871,
                            "top_k_retrieval": 42,
                            "topk_20": 0.6,
                            "topk_retrieval_20": 12,
                            "true_label": 67,
                        },
                        "Patient007": {
                            "topk": 0.4117647058823529,
                            "roc": 0.4498663101604278,
                            "logloss": 19.483678857810958,
                            "precision": 0.38095238095238093,
                            "recall": 0.47058823529411764,
                            "f1": 0.42105263157894735,
                            "top_k_retrieval": 14,
                            "topk_20": 0.45,
                            "topk_retrieval_20": 9,
                            "true_label": 34,
                        },
                        "global": {
                            "topk_retrieval_20_patientid": 50,
                            "topk_retrieval_patientid": 133,
                            "topk_20_patientid": 0.5,
                            "topk_patientid": 0.5341365461847389,
                        },
                    },
                    "global": {
                        "topk": 0.5381526104417671,
                        "roc": 0.5360885774172387,
                        "logloss": 15.82180701465919,
                        "precision": 0.5372549019607843,
                        "recall": 0.5502008032128514,
                        "f1": 0.5436507936507935,
                        "topk_retrieval_20_patientid": 50,
                        "topk_retrieval_patientid": 133,
                        "topk_20_patientid": 0.5,
                        "topk_patientid": 0.5341365461847389,
                    },
                }
            }
        },
    }


@pytest.fixture(scope="session")
def best_experiment_id() -> List[str]:
    """Dummy best experiment id."""
    return ["KfoldExperiment", "Fspca", "XgboostModel"]


@pytest.fixture(scope="session")
def generate_peptide_allele_output() -> str:
    """Dummy generate pepetide allele pair path."""
    return "tests/fixtures/generated_pepetide_allele.csv"


@pytest.fixture(scope="session")
def expression_dummy_data():
    """Expression dummy data."""
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "gene_expression_1": [0.5, 0.8, 0.2],
            "second_id": [0, 1, 2],
            "target": [1, 0, 1],
            "rnalocalization": [1, 0, 1],
        }
    )
    return data


@pytest.fixture(scope="session")
def preprocess_train_df() -> pd.DataFrame:
    """Sample dataframe used to test process_train function."""
    return pd.DataFrame({"A": [0, 1, 0, 1, 0, 5], "diff_class_prot": [1, 2, 3, 4, 5, 0]})


@pytest.fixture(scope="session")
def mock_data_dict(preprocess_train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Sample dictionary containing train and test dataframes."""
    return {
        "train": preprocess_train_df.assign(
            **{
                "col1": [1, 2, 3, 7, 8, 9],
                "col2": [1.2, 3.4, 5.6, 1.2, 3.5, 2.6],
                "col3": ["d", "e", "f", "a", "b", "c"],
                "col4": [False, True, False, False, True, False],
            }
        ),
        "test": pd.DataFrame(
            {
                "col1": [4, 5, 6, 7, 8, 9],
                "col2": [1.2, 3.4, 5.6, 1.2, 3.5, 2.6],
                "col3": ["d", "e", "f", "a", "b", "c"],
                "col4": [False, True, False, False, True, False],
            }
        ),
    }


@pytest.fixture(scope="session")
def mock_features() -> List[str]:
    """Sample mock features."""
    return ["col1", "col2", "col3", "col4"]


@pytest.fixture(scope="session")
def mock_data():
    """Sample data for integration tests."""
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score1": [0.5, 0.8, 0.3],
            "score2": [0.2, 0.4, 0.6],
            "raw_expression": ["A", "B", "C"],
            "feature1": [10, 20, 30],
            "feature2": [True, False, True],
            "feature3": ["X", "Y", "Z"],
            "target": [0, "NA", 1],
        }
    )
    return data


@pytest.fixture(scope="session")
def mock_configuration():
    """Sample mock configuration."""
    configuration = MagicMock()
    configuration.label = "target"
    configuration.proxy_scores = {"score1": "renamed_score1", "score2": "renamed_score2"}
    configuration.expression = {"raw_name": "expression_raw", "name": "expression_column"}
    configuration.expression_filter = "exp"
    configuration.expression_name = "expression_column"
    configuration.include_features = ["feature1", "feature2"]
    return configuration


@pytest.fixture(scope="session")
def raw_data_paths() -> str:
    """Dummy raw train data."""
    path = "tests/fixtures/raw_data.csv"
    return path


@pytest.fixture(scope="session")
def processing_config_path() -> str:
    """Dummy processing config path."""
    path = "processing_configuration.yml"
    return path


@pytest.fixture(scope="session")
def processed_dummy_data_path(processing_config_path) -> Path:
    """Processed_dummy_data_path."""
    configuration = load_yml(CONFIGURATION_DIRECTORY / processing_config_path)
    data_version = configuration["data_version"]
    data_type = configuration["data_type"]
    processed_data_path = DATA_DIRECTORY / f"proc_data_{data_version}_{data_type}"
    return processed_data_path


@pytest.fixture(scope="session")
def main_raw_data_paths() -> str:
    """Dummy raw main data."""
    path = ("tests/fixtures/dummy_raw_data.csv",)
    return path
