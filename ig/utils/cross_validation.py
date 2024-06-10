"""Module used to define some helper functions for running experiments."""
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from ig import EXP_NAMES, EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY, FEATURES_SELECTION_DIRECTORY
from ig.utils.io import load_yml
from ig.utils.logger import get_logger

log = get_logger("utils/experiments")


def import_experiment(base_module: ModuleType, target_class_name: str) -> Callable:
    """Import class from a submodule."""
    for module_name, module in inspect.getmembers(base_module):

        if inspect.ismodule(module):
            for class_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) & (class_name == target_class_name):

                    model = getattr(base_module, module_name)
                    model = getattr(model, target_class_name)
                    return model
    raise NotImplementedError(
        f" {target_class_name} Class is not implemented in {base_module.__name__}"
    )


def get_model_by_name(base_module: ModuleType, target_class_name: str) -> Callable:
    """Import a class from module."""
    for class_name, obj in inspect.getmembers(base_module):
        if inspect.isclass(obj) & (class_name == target_class_name):
            model = getattr(base_module, target_class_name)
            return model
    raise NotImplementedError(
        f" {target_class_name} Class is not implemented in {base_module.__name__}"
    )


def maybe_int(variable: Any) -> int:
    """Cast a variable to int."""
    try:
        variable = int(variable)
    except ValueError:
        pass

    return variable


def load_experiments(config: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load experiments from a configuration file."""
    experiments = config["experiments"]
    experiment_names = []
    for exp_name in experiments.keys():
        if exp_name in EXP_NAMES:
            experiment_names.append(exp_name)
        else:
            log.info("Experiment with the name %s is not available", exp_name)
    return experiment_names, [experiments[exp_name] for exp_name in experiment_names]


def load_models(
    general_configuration: Dict[str, Any], experiment_path: Path
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load the available models from the configuration file."""
    models_configuration_directory = experiment_path / EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY
    models = general_configuration["models"]
    model_types = []
    model_config = []
    for config_path in models:
        model = load_yml(models_configuration_directory / config_path)
        model_types.append(model["model_type"])
        model_config.append(model["model_config"])
    return model_types, model_config


def plotting_shap_values(
    shap_values: Any, data: pd.DataFrame, features: List[str], fig_name: Union[str, Path]
) -> None:
    """Plot single experiment shap values."""
    shap.summary_plot(
        shap_values, data[features].values, feature_names=features, plot_type="violin", show=False
    )
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    plt.clf()


def plotting_kfold_shap_values(df: pd.DataFrame, fig_name: Union[str, Path]) -> None:
    """Plotting kfold shap values mean resuls."""
    plt.figure()
    plt.title("Kfold feature importance")
    df.sort_values(by=["scores"], ascending=False, inplace=True)
    snsplot = sns.barplot(x=df.scores, y=df.features)
    snsplot.figure.savefig(fig_name, dpi=200, bbox_inches="tight")


def save_features(features: List[str], file_path: Path) -> None:
    """Save a list of features as a text format."""
    file_path = file_path / "features.txt"
    with open(file_path, "w") as f:
        for e in features:
            f.write(str(e) + "\n")


def load_features(file: Path) -> List[str]:
    """Load a list of features previously saved in text format ."""
    if not file.suffix:
        file = file.parent / (file.name + ".txt")
    if not file.exists():
        raise FileExistsError(
            f"feature list {file.name}  does not exist"
            f"in the {FEATURES_SELECTION_DIRECTORY} folder please check the file name."
        )
    with open(file) as f:
        features = [line.rstrip() for line in f]
        return features


def is_number(s: Any) -> bool:
    """Check if s is a number."""
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_int_params(names: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert params to int."""
    for int_type in names:
        # sometimes the parameters can be choices between options or numerical values. like "log2" vs "1-10"
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params
