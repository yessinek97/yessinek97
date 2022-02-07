"""Init."""
from pathlib import Path
from typing import Any
from typing import Dict

Evals = Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
SINGLE_MODEL_NAME = "SingleModel"
KFOLD_MODEL_NAME = "KfoldExperiment"
DKFOLD_MODEL_NAME = "DoubleKfold"
CURRENT_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = CURRENT_DIRECTORY.parent


CONFIGURATION_DIRECTORY = ROOT_DIRECTORY / "configuration"
FEATURES_DIRECTORY = CONFIGURATION_DIRECTORY / "features"
MODELS_DIRECTORY = ROOT_DIRECTORY / "models"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
MODEL_CONFIGURATION_DIRECTORY = CONFIGURATION_DIRECTORY / "model_config"
FS_CONFIGURATION_DIRECTORY = CONFIGURATION_DIRECTORY / "FS_config"
DATAPROC_DIRACTORY = "data_proc"
FEATURIZER_DIRECTORY = "featurizer"
ID_NAME = "ID_column"
