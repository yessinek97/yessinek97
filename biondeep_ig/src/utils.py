"""Module de define some helper functions."""
import inspect
import json
import pickle
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import yaml

from biondeep_ig.src import FEATURES_DIRECTORY
from biondeep_ig.src import FS_CONFIGURATION_DIRECTORY
from biondeep_ig.src import MODEL_CONFIGURATION_DIRECTORY
from biondeep_ig.src import MODELS_DIRECTORY
from biondeep_ig.src.logger import get_logger

log = get_logger("utils")


def save_yml(data, file_path):
    """Save a dictionary to a yaml format."""
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yml(file_path):
    """Load a yaml file to a Python dictionary."""
    with open(file_path, "r") as f:
        data = yaml.full_load(f)

    return data


def save_as_pkl(obj, file_path):
    """Save a python object as a pickle."""
    with file_path.open("wb") as f:
        pickle.dump(obj, f)


def load_pkl(file_path):
    """Load a pickle previously saved by save_as_pkl function."""
    with file_path.open("rb") as f:
        obj = pickle.load(f)

    return obj


def load_json(file_path):
    """Load a json previously saved by save_as_json function."""
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def save_as_json(data, file_path):
    """Save a dictionary as json file.

    Args:
        data: dictionary to save
        file_path: path to save the file
        save_kwargs: kwargs to forward to json.dump method.
            By default 'indent' is set to 4 for a better display.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


def import_experiment(base_module, target_class_name):
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


def get_model_by_name(base_module, target_class_name):
    """Import a class from module."""
    for class_name, obj in inspect.getmembers(base_module):
        if inspect.isclass(obj) & (class_name == target_class_name):
            model = getattr(base_module, target_class_name)
            return model
    raise NotImplementedError(
        f" {target_class_name} Class is not implemented in {base_module.__name__}"
    )


def maybe_int(variable):
    """Cast a varible to int."""
    try:
        variable = int(variable)
    except ValueError:
        pass

    return variable


def load_experiments(config):
    """Load experiments from a configuration file."""
    experiments = config["experiments"]
    return list(experiments.keys()), list(experiments.values())


def load_models(config):
    """Load the available models from the configuration file."""
    models = config["models"]
    model_types = []
    model_config = []
    for config_path in models:
        model = load_yml(MODEL_CONFIGURATION_DIRECTORY / config_path)
        model_types.append(model["model_type"])
        model_config.append(model["model_config"])
    return model_types, model_config


def load_fs(config):
    """Load the available fs methods from the configuration file."""
    fs = config["FS"]["FS_methods"]
    fs_types = []
    fs_config = []
    for config_path in fs:
        fs = load_yml(FS_CONFIGURATION_DIRECTORY / "FS_method_config" / config_path)
        fs_types.append(fs["FS_type"])
        fs_config.append(fs["FS_config"])
    return fs_types, fs_config


def _create_dict_for_benchmark_scores_fold_method(
    validation_metrics, test_metrics, benchmark_column
):
    """Safely extract the benchmark scores in dict for kfold method."""
    benchmark_test_score = {}
    benchmark_val_score = {}
    for col in benchmark_column:
        benchmark_val_score[f"val_{col}"] = (
            validation_metrics[validation_metrics.fold == col].iloc[0].topk
        )
        benchmark_test_score[f"test_{col}"] = test_metrics[test_metrics.fold == col].iloc[0].topk
    return benchmark_val_score, benchmark_test_score


def _create_dict_for_benchmark_scores_single_model_method(
    validation_metrics, test_metrics, benchmark_column
):
    """Safely extract the benchmark scores in dict for single model method."""
    benchmark_test_score = {}
    benchmark_val_score = {}
    for col in benchmark_column:
        benchmark_val_score[f"val_{col}"] = validation_metrics.iloc[0].topk
        benchmark_test_score[f"test_{col}"] = test_metrics.iloc[0].topk
    return benchmark_val_score, benchmark_test_score


def get_best_experiment(
    results: pd.DataFrame,
    eval_configuration: Dict[str, Any],
    path: Path,
    file_name: str = "results",
) -> Tuple[str, str]:
    """Return the best experiment."""
    display = ""
    metric_selector = eval_configuration.get("metric_selector", "topk")
    monitoring_metrics = eval_configuration.get("monitoring_metrics", [metric_selector])
    validation = pd.concat([result["validation"] for result in results])
    test = pd.concat([result["test"] for result in results])
    pd.concat([validation, test]).to_csv(path / f"{file_name}.csv", index=False)
    statistic_list = []
    for result in results:
        try:
            statistic_list.append(result["statistic"])
        except KeyError:
            pass
    statistic = pd.DataFrame(statistic_list)
    data_selector = pd.concat(
        [result[eval_configuration["data_name_selector"]] for result in results]
    )
    data_selector.sort_values(
        [metric_selector], ascending=eval_configuration["metrics_selector_higher"], inplace=True
    )

    best_experiment_id = data_selector.iloc[-1]["ID"]

    best_validation = validation[validation.ID == best_experiment_id]
    best_test = test[test.ID == best_experiment_id]

    display += "###### Best run ######\n"
    display += f"Experiment : {best_test.experiment.iloc[0]} \n"
    display += f"model :  {best_test.model.iloc[0]}\n"
    display += f"features :  {best_test.features.iloc[0]}\n"
    display += f"seed :  {best_test.seed.iloc[0]}\n" if "seed" in best_test.columns else ""
    display += (
        f"nbr of folds :  {best_test.nbr_fold.iloc[0]}\n" if "nbr_fold" in best_test.columns else ""
    )
    display += f"prediction :  {best_test.prediction.iloc[0]}\n"
    validation_metrics_display = " ".join(
        [f"{metric} : {best_validation[metric].iloc[0]:.3f}" for metric in monitoring_metrics]
    )
    display += f"Validation score :  {validation_metrics_display}\n"
    test_metrics_display = " ".join(
        [f"{metric} : {best_test[metric].iloc[0]:.3f}" for metric in monitoring_metrics]
    )
    display += f"Test score :  {test_metrics_display}\n"
    if len(statistic):
        beststatistic_list = statistic[statistic.ID == best_experiment_id]
        if len(beststatistic_list):
            for e in beststatistic_list.drop(["ID"], axis=1).columns:
                display += f"{e} :  {beststatistic_list[e].iloc[0]}\n"
    return best_experiment_id, display


def wrap(pre, post):
    """Wrapper."""

    def decorate(func):
        """Decorator."""

        def call(*args, **kwargs):
            """Actual wrapping."""
            pre(func, args[0].folder_name, args[0].experiment_name, args[0].model_type)
            result = func(*args, **kwargs)
            post(func)
            return result

        return call

    return decorate


def entering(func, folder_name, experiment_name, model_type):
    """Pre function logging."""
    path_log = MODELS_DIRECTORY / folder_name / "models.log"
    with open(path_log, "a+") as f:
        f.write("**********************************")
        f.write(experiment_name)
        f.write("----")
        f.write(model_type)
        f.write("**********************************")
        f.write("\n")
    sys.stdout = open(path_log, "a")
    log.debug("Entered %s", func.__name__)


def exiting(func):
    """Post function logging."""
    # sys.stdout.close()
    log.debug("Exited  %s", func.__name__)


def is_number(s):
    """Check if s is a number."""
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_int_params(names, params):
    """Convert params to int."""
    for int_type in names:
        # sometimes the parameters can be choices between options or numerical values. like "log2" vs "1-10"
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params


def plotting_shap_values(shap_values, data, features, fig_name):
    """Plot single experiment shap values."""
    shap.summary_plot(
        shap_values, data[features].values, feature_names=features, plot_type="violin", show=False
    )
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    plt.clf()


def plotting_kfold_shap_values(df, fig_name):
    """Plotting kfold shap values mean resuls."""
    plt.figure()
    plt.title("Kfold feature importance")
    df.sort_values(by=["scores"], ascending=False, inplace=True)
    snsplot = sns.barplot(x=df.scores, y=df.features)
    snsplot.figure.savefig(fig_name, dpi=200, bbox_inches="tight")


def save_features(features, file_path):
    """Save a list of features as a text format."""
    file_path = file_path / "features.txt"
    with open(file_path, "w") as f:
        for e in features:
            f.write(str(e) + "\n")


def load_features(file):
    """Load a list of features previously saved in text format ."""
    if not file.suffix:
        file = file.parent / (file.name + ".txt")

    if not file.exists():
        raise FileExistsError(
            (
                f"feature list {file.name}  does not exist"
                "in the configuration/features folder please check the file name."
            )
        )
    with open(file, "r") as f:
        features = [line.rstrip() for line in f]
        return features


def log_summary_results(display):
    """Print summary results."""
    for line in display.split("\n"):
        log.info(line)


def get_task_name(label_name):
    """Return the folder name to the correspending label name."""
    if label_name.lower() == "cd4_any":
        return "CD4"
    if label_name.lower() == "cd8_any":
        return "CD8"
    raise ValueError("label name not known in fs ...")


def remove_genrated_features(features_names, label_name):
    """Remove Genrated features files by feature selection process."""
    task = get_task_name(label_name)
    for features_name in features_names:
        (FEATURES_DIRECTORY / task / f"{features_name}.txt").unlink()


def read(file_path: str, **kwargs):
    """Read data."""
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
