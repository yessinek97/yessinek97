"""Module de define some helper functions."""
import inspect
import json
import logging
import os
import pickle
import random
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml

from ig import (
    DATA_DIRECTORY,
    DATAPROC_DIRECTORY,
    DEFAULT_SEED,
    EXP_NAMES,
    EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY,
    FEATURES_DIRECTORY,
    FEATURES_SELECTION_DIRECTORY,
    FS_CONFIGURATION_DIRECTORY,
    MAX_RANDOM_SEED,
    MODEL_CONFIGURATION_DIRECTORY,
)
from ig.src.logger import get_logger

log = get_logger("utils")


def save_yml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save a dictionary to a yaml format."""
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a yaml file to a Python dictionary."""
    with open(file_path) as f:
        data = yaml.full_load(f)

    return data


def save_as_pkl(obj: Any, file_path: Union[str, Path]) -> None:
    """Save a python object as a pickle."""
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file_path: Union[str, Path]) -> Any:
    """Load a pickle previously saved by save_as_pkl function."""
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    return obj


# Loads an embeddings pickle file for probing experiment
def load_embedding_file(file_path: str) -> Any:
    """Load pre generated sequence embeddings from a pickle file.

    Args:
        file_path (str): Path to the embedding file

    Raises:
        FileNotFoundError: raise an error if the file can't be found

    Returns:
        Any: return an object with the file
    """
    local_path = DATA_DIRECTORY / file_path
    if Path(local_path).exists():
        log.info("Loading embedding pickle file at: %s", local_path)
        return load_pkl(local_path)
    raise FileNotFoundError("The embeddings pickle file %s cannot be found!!!" % local_path)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a json previously saved by save_as_json function."""
    with open(file_path) as json_file:
        return json.load(json_file)


def save_as_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save a dictionary as json file.

    Args:
        data: dictionary to save
        file_path: path to save the file
        save_kwargs: kwargs to forward to json.dump method.
            By default 'indent' is set to 4 for a better display.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


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


def copy_models_configuration_files(
    general_configuration: Dict[str, Any], experiment_path: Path
) -> None:
    """Copy model configuration file to  the experiment folder."""
    models_configuration_directory = experiment_path / EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY
    models_configuration_directory.mkdir(exist_ok=True, parents=True)
    for models_configuration_name in general_configuration["models"]:
        shutil.copyfile(
            MODEL_CONFIGURATION_DIRECTORY / models_configuration_name,
            models_configuration_directory / models_configuration_name,
        )


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


def copy_embeddings_file(source_path: Path, destination_path: Path) -> bool:
    """Save embeddings file when doing probing to the current experiment directory.

    Args:
        source_path (Path): source embedding file path
        destination_path (Path): destination path inside the current data_proc directory

    Returns:
        bool: retrun true if the file was copied correctly
    """
    # shutil.copy2() method copies the source file to the destination directory with it's metadata
    dst = shutil.copy2(source_path, destination_path)

    return dst.exists()


def change_embeddings_path(config_path: Path, new_path: Path) -> bool:
    """Change emb_file_path in llm models config file to the new file that was copyed inside the experiment's data_proc directory.

    Args:
        config_path (Path): llm models config file path
        new_path (Path): the path of the copied embeddings file

    Returns:
        bool: return true if the change was successfull
    """
    config = load_yml(config_path)

    config["model_config"]["general_params"]["emb_file_path"] = str(new_path)

    save_yml(config, config_path)

    return config_path.exists()


def save_probing_embeddings(experiment_path: Path, emb_file_path: Path) -> None:
    """Check if the current experiment has an LLM probing model then save it's embeddings file to the models directory.

    Args:
        experiment_path (Path): Path to the current experiment
        emb_file_path (Path): Path to the probing embeddings file

    Returns:
        None
    """
    emb_file_path = DATA_DIRECTORY / emb_file_path
    file_name = Path(emb_file_path).name
    emb_dest_path = experiment_path / DATAPROC_DIRECTORY / file_name

    # Copy the embeddings file to data_proc/ inside the current experiment directory
    saved = copy_embeddings_file(emb_file_path, emb_dest_path)

    # Change the embeddings path in current experiment llm model config
    llm_model_conf_path = (
        experiment_path / EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY / "llm_based_probing_config.yml"
    )
    changed = change_embeddings_path(llm_model_conf_path, emb_dest_path)

    if saved & changed:
        log.info("Embeddings file was successfully copied to : %s", emb_dest_path)
    else:
        log.warning("The embeddings file for probing experiment was not saved!!!")


def load_fs(config: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load the available fs methods from the configuration file."""
    fs = config["FS"]["FS_methods"]
    fs_types = []
    fs_config = []
    for config_path in fs:
        fs = load_yml(FS_CONFIGURATION_DIRECTORY / "FS_method_config" / config_path)
        fs_types.append(fs["FS_type"])
        fs_config.append(fs["FS_config"])
    return fs_types, fs_config


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
    if len(statistic_list) > 0:
        statistic = pd.concat(statistic_list)
        beststatistic_list = statistic[statistic.ID == best_experiment_id]
        if len(beststatistic_list):
            for e in beststatistic_list.drop(["ID"], axis=1).columns:
                display += f"{e} :  {beststatistic_list[e].iloc[0]}\n"
    return best_experiment_id, display


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


def log_summary_results(display: str) -> None:
    """Print summary results."""
    for line in display.split("\n"):
        log.info(line)


def get_task_name(label_name: str) -> str:
    """Return the folder name to the corresponding label name."""
    if label_name.lower() == "cd4_any":
        return "CD4"
    if label_name.lower() == "cd8_any":
        return "CD8"
    raise ValueError("label name not known in fs ...")


def read_data(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """This function is used to read the input dataset.

    Args:
        file_path: Path of the input dataset.

    Returns:
        df: The input dataframe.
    """
    logging.info("Loading: %s", file_path)

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


def copy_existing_features_lists(
    feature_list: List[str], experiment_path: Path, label_name: str
) -> None:
    """Move features list from features directory to the experiment directory."""
    original_directory = FEATURES_DIRECTORY / get_task_name(label_name=label_name)
    features_selection_directory = experiment_path / FEATURES_SELECTION_DIRECTORY
    features_selection_directory.mkdir(exist_ok=True, parents=True)
    for feature in feature_list:
        shutil.copyfile(
            original_directory / f"{feature}.txt",
            experiment_path / FEATURES_SELECTION_DIRECTORY / f"{feature}.txt",
        )


def seed_basic(seed: int) -> None:
    """Seed every thing."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def remove_bucket_prefix(uri: str, keep: Optional[int] = -2) -> str:
    """Remove the bucket prefix and the bucket name from the file path in order to keep the relative path.

    Args:
        uri: uri to clean
        keep: int
    >>> remove_bucket_prefix("gs://biondeep-data/IG/data/Ig_2022/train.csv",-2)
    'Ig_2022/train.csv'
    >>> remove_bucket_prefix("gs://biondeep-data/IG/data/Ig_2022/train.csv",-1)
    'train.csv'
    """
    return os.path.join(*uri.split("/")[keep:])


def generate_random_seeds(nbr_seeds: int) -> List[int]:
    """Generate a list of random values between 0 and MAX_RANDOM_SEED with length equal to nbr_seeds."""
    random_seeds: List[int] = []
    random.seed(DEFAULT_SEED)
    while len(random_seeds) < nbr_seeds:
        random_seeds = list(set(random_seeds + [random.randint(0, MAX_RANDOM_SEED)]))
    return random_seeds


def check_and_create_folder(experiment_path: Path) -> None:
    """Check if the checkpoint folder  exists or not."""
    if experiment_path.exists():
        click.confirm(
            (
                f"The  folder with the name {experiment_path.name} already exists."
                "Do you want to continue but"
                "all the files will be deleted?"
            ),
            abort=True,
        )
        shutil.rmtree(experiment_path)

    experiment_path.mkdir(exist_ok=True, parents=True)


def crop_sequences(
    sequences: List[str], mutation_start_positions: List[int], context_length: int
) -> List[str]:
    """Crops the sequences to match desired context length."""
    sequences = [
        seq[
            max(0, mut_pos - context_length // 2) : max(0, mut_pos - context_length // 2)
            + context_length
        ]
        for seq, mut_pos in zip(sequences, mutation_start_positions)
    ]
    return sequences
