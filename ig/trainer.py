"""Module used to define the training command."""
import copy
import shutil
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd

import ig.src.experiments as exper
from ig import (
    CONFIGURATION_DIRECTORY,
    DATAPROC_DIRECTORY,
    DEFAULT_SEED,
    EXP_NAMES,
    KFOLD_EXP_NAMES,
    KFOLD_MODEL_NAME,
    KFOLD_MULTISEED_MODEL_NAME,
    MODELS_DIRECTORY,
    SINGLE_MODEL_NAME,
)
from ig.bucket.click import arguments
from ig.constants import EvalExpType, MetricsEvalType, TuneResults
from ig.dataset.dataset import Dataset
from ig.feature_selection import feature_selection_main
from ig.src.evaluation import Evaluation
from ig.src.experiments.tuning import Tuning
from ig.src.logger import NeptuneLogs, get_logger, init_logger
from ig.src.metrics import topk_global
from ig.src.utils import (
    copy_existing_features_lists,
    copy_models_configuration_files,
    generate_random_seeds,
    get_best_experiment,
    import_experiment,
    load_experiments,
    load_models,
    load_yml,
    log_summary_results,
    save_yml,
    seed_basic,
)

log: Logger = get_logger("Train")


@click.command()
@click.option(
    "--train_data_path",
    "-train",
    type=str,
    required=True,
    help="Path to the dataset used in training",
)
@click.option(
    "--test_data_path",
    "-test",
    type=str,
    required=True,
    help="Path to the dataset used in  evaluation.",
)
@click.option(
    "--unlabeled_path",
    "-unlabeled",
    type=str,
    required=False,
    help="Path to the unlabeled dataset.",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
@arguments.force(help="Force overwrite the local file if it already exists.")  # type: ignore
@click.pass_context
def train(
    ctx: Union[click.core.Context, Any],
    train_data_path: str,
    test_data_path: str,
    unlabeled_path: str,
    configuration_file: str,
    folder_name: str,
    force: bool,
    multi_train: bool = False,
) -> None:
    """Launch the training of a model based on the provided configuration and the input training file."""
    if multi_train:
        experiment_path = Path(folder_name)
        general_configuration = load_yml(configuration_file)

        experiment_path.mkdir(parents=True, exist_ok=True)
    else:
        experiment_path = MODELS_DIRECTORY / folder_name
        _check_model_folder(experiment_path)
        general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    init_logger(logging_directory=experiment_path)
    save_yml(general_configuration, str(experiment_path / "run_configuration.yml"))
    seed_basic(DEFAULT_SEED)
    log.info("Started")
    eval_configuration: Dict[str, str] = general_configuration["evaluation"]
    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(general_configuration)
    log.info("****************************** Load EXP ****************************** ")
    copy_models_configuration_files(
        general_configuration=general_configuration, experiment_path=experiment_path
    )

    model_types, model_params = load_models(
        general_configuration=general_configuration, experiment_path=experiment_path
    )
    log.info("****************************** Load Models ****************************** ")

    train_data, test_data = load_datasets(
        click_ctx=ctx,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        general_configuration=general_configuration,
        experiment_path=experiment_path,
        force=force,
    )

    log.info("*********************** Load and process Train , Test *********************** ")
    existing_features_lists: List[str] = general_configuration.get("feature_paths", [])
    if "FS" in general_configuration:

        features_names = feature_selection_main(
            train_data=train_data,
            general_configuration=general_configuration,
            folder_name=folder_name,
            with_train=True,
        )
        log.info("features_names : %s", " ".join(features_names))
        log.info("****************************** Finished FS ****************************** ")

        feature_paths = existing_features_lists + features_names
        general_configuration["feature_paths"] = feature_paths

    copy_existing_features_lists(
        existing_features_lists,
        experiment_path=experiment_path,
        label_name=general_configuration["label"],
    )
    save_yml(general_configuration, str(experiment_path / "configuration.yml"))
    displays: str = "####### Runs Summary #######"
    results: List[EvalExpType] = []
    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["simple train"], training_path=train_data_path, test_path=test_data_path
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)
    comparison_score_metrics = eval_comparison_score(
        configuration=general_configuration,
        train_data=train_data().copy(),
        test_data=test_data().copy(),
        dataset_name=test_data.dataset_name,
        experiment_path=experiment_path,
        plot_comparison_score_only=True,
    )
    for experiment_name, experiment_param in zip(experiment_names, experiment_params):
        log.info("%s :", experiment_name)
        configuration = copy.deepcopy(general_configuration)
        configuration.pop("models")
        displays += f"\n Method :{experiment_name}: "
        for model_type, model_param in zip(model_types, model_params):
            log.info("  %s:", model_type)
            result, display = _train_func(
                experiment_name=experiment_name,
                experiment_param=experiment_param,
                model_type=model_type,
                model_param=model_param,
                train_data=train_data,
                test_data=test_data,
                unlabeled_path=unlabeled_path,
                configuration=configuration,
                folder_name=folder_name,
                log_handler=log,
                neptune_log=neptune_log,
                comparison_score_metrics=comparison_score_metrics,
            )
            results.extend(result)
            displays += display
    log_summary_results(displays)
    best_experiment_id, eval_message = get_best_experiment(
        results, eval_configuration, path=experiment_path, file_name="results"
    )
    log_summary_results(eval_message)
    best_experiment_id_splitted = best_experiment_id.split("//")
    neptune_log.upload_experiment(
        experiment_path=(
            experiment_path
            / best_experiment_id_splitted[0]
            / best_experiment_id_splitted[1]
            / best_experiment_id_splitted[2]
        ),
        neptune_sub_folder="best_experiment",
    )
    _save_best_experiment(
        best_experiment_id=best_experiment_id_splitted, experiment_path=experiment_path
    )
    remove_unnecessary_folders(experiment_path, train_data.remove_unnecessary_folders)


@click.command()  # noqa
@click.option(
    "--train_data_path",
    "-train",
    type=str,
    required=True,
    help="Path to the dataset used in training",
)
@click.option(
    "--test_data_path",
    "-test",
    type=str,
    required=True,
    help="Path to the dataset used in  evaluation.",
)
@click.option(
    "--unlabeled_path",
    "-unlabeled",
    type=str,
    required=False,
    help="Path to the unlabeled dataset.",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
@click.pass_context
def train_seed_fold(  # noqa
    ctx: Union[click.core.Context, Any],
    train_data_path: str,
    test_data_path: str,
    unlabeled_path: str,
    configuration_file: str,
    folder_name: str,
) -> None:
    """Train with multiple seed and folds."""
    experiment_path = MODELS_DIRECTORY / folder_name
    _check_model_folder(experiment_path)
    init_logger(logging_directory=experiment_path)
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    copy_existing_features_lists(
        general_configuration.get("feature_paths", []),
        experiment_path=experiment_path,
        label_name=general_configuration["label"],
    )
    experiment_names, experiment_params = load_experiments(general_configuration)
    copy_models_configuration_files(
        general_configuration=general_configuration, experiment_path=experiment_path
    )
    model_types, model_params = load_models(
        general_configuration=general_configuration, experiment_path=experiment_path
    )
    seeds: List[int] = general_configuration["processing"]["seeds"]
    folds: List[int] = general_configuration["processing"]["folds"]
    eval_configuration: Dict[str, str] = general_configuration["evaluation"]

    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["Multi seed fold train"], training_path=train_data_path, test_path=test_data_path
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)
    save_yml(general_configuration, experiment_path / "configuration.yml")
    log.info("****************************** Load YAML ****************************** ")
    log.info("****************************** Load EXP ****************************** ")
    log.info("****************************** Load Models ****************************** ")
    displays: str = "####### Runs Summary #######"
    results: List[EvalExpType] = []
    # pylint: disable=too-many-nested-blocks
    for experiment_name, experiment_param in zip(experiment_names, experiment_params):
        log.info("%s :", experiment_name)
        configuration = copy.deepcopy(general_configuration)
        configuration.pop("models")
        displays += f"\n Method :{experiment_name}: "

        for model_type, model_param in zip(model_types, model_params):
            log.info("  -%s :", model_type)
            for seed in seeds:
                log.info("      -seed %s :", seed)
                configuration["processing"]["seed"] = seed
                if experiment_name == "SingleModel":
                    train_data, test_data = load_datasets(
                        click_ctx=ctx,
                        train_data_path=train_data_path,
                        test_data_path=test_data_path,
                        general_configuration=configuration,
                        experiment_path=experiment_path,
                        force=False,
                    )
                    result, display = _train_func(
                        experiment_name=experiment_name,
                        experiment_param=experiment_param,
                        model_type=model_type,
                        model_param=model_param,
                        train_data=train_data,
                        test_data=test_data,
                        unlabeled_path=unlabeled_path,
                        configuration=configuration,
                        folder_name=folder_name,
                        sub_folder_name=f"seed_{seed}",
                        neptune_log=neptune_log,
                        log_handler=log,
                    )

                    for e in result:
                        e["test"]["seed"] = seed
                        e["validation"]["seed"] = seed
                        e["test"]["ID"] = e["test"].ID + f"//seed_{seed}"
                        e["validation"]["ID"] = e["validation"].ID + f"//seed_{seed}"
                    results.extend(result)

                    displays += f"\n -Seed : {seed}" + display
                else:
                    for fold in folds:
                        log.info("         -fold : %s:", fold)
                        configuration["processing"]["fold"] = fold
                        train_data, test_data = load_datasets(
                            click_ctx=ctx,
                            train_data_path=train_data_path,
                            test_data_path=test_data_path,
                            general_configuration=configuration,
                            experiment_path=experiment_path,
                            force=False,
                        )
                        result, display = _train_func(
                            experiment_name=experiment_name,
                            experiment_param=experiment_param,
                            model_type=model_type,
                            model_param=model_param,
                            train_data=train_data,
                            test_data=test_data,
                            unlabeled_path=unlabeled_path,
                            configuration=configuration,
                            folder_name=folder_name,
                            sub_folder_name=f"seed_{seed}_fold_{fold}",
                            neptune_log=neptune_log,
                            log_handler=log,
                        )
                        for e in result:
                            e["test"]["seed"] = int(seed)
                            e["test"]["nbr_fold"] = int(fold)
                            e["test"]["ID"] = e["test"]["ID"].iloc[0] + f"//seed_{seed}_fold_{fold}"
                            e["validation"]["seed"] = int(seed)
                            e["validation"]["nbr_fold"] = int(fold)
                            e["validation"]["ID"] = (
                                e["validation"]["ID"].iloc[0] + f"//seed_{seed}_fold_{fold}"
                            )
                            e["statistic"]["ID"] = (
                                e["statistic"]["ID"] + f"//seed_{seed}_fold_{fold}"
                            )
                        results.extend(result)

                        displays += f"\n -Seed : {seed}  -Fold : {fold}" + display

    log_summary_results(displays)
    best_experiment_id, eval_message = get_best_experiment(
        results, eval_configuration, path=experiment_path, file_name="results"
    )
    log_summary_results(eval_message)
    best_experiment_id_splited = best_experiment_id.split("//")
    neptune_log.upload_experiment(
        experiment_path=(
            experiment_path
            / best_experiment_id_splited[0]
            / best_experiment_id_splited[1]
            / best_experiment_id_splited[3]
            / best_experiment_id_splited[2]
        ),
        neptune_sub_folder="best_experiment",
    )
    _save_best_experiment(
        best_experiment_id=best_experiment_id_splited, experiment_path=experiment_path
    )
    remove_unnecessary_folders(experiment_path, train_data.remove_unnecessary_folders)


@click.command()  # noqa
@click.option(
    "--train_data_path",
    "-train",
    type=str,
    required=True,
    help="Path to the dataset used in training",
)
@click.option(
    "--test_data_path",
    "-test",
    type=str,
    required=True,
    help="Path to the dataset used in  evaluation.",
)
@click.option(
    "--unlabeled_path",
    "-unlabeled",
    type=str,
    required=False,
    help="Path to the unlabeled dataset.",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
@arguments.force(help="Force overwrite the local file if it already exists.")  # type: ignore
@click.pass_context
def tune(
    ctx: Union[click.core.Context, Any],
    train_data_path: str,
    test_data_path: str,
    unlabeled_path: str,
    configuration_file: str,
    folder_name: str,
    force: bool,
) -> None:
    """Tune model params."""
    experiment_path = MODELS_DIRECTORY / folder_name
    _check_model_folder(experiment_path)
    init_logger(logging_directory=experiment_path)
    log.info("Started")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    copy_existing_features_lists(
        general_configuration.get("feature_paths", []),
        experiment_path=experiment_path,
        label_name=general_configuration["label"],
    )
    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(general_configuration)
    log.info("****************************** Load EXP ****************************** ")
    copy_models_configuration_files(
        general_configuration=general_configuration, experiment_path=experiment_path
    )
    model_types, model_params = load_models(
        general_configuration=general_configuration, experiment_path=experiment_path
    )
    log.info("****************************** Load Models ****************************** ")
    results: List[TuneResults] = []
    save_yml(general_configuration, experiment_path / "configuration.yml")
    train_data, test_data = load_datasets(
        click_ctx=ctx,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        general_configuration=general_configuration,
        experiment_path=experiment_path,
        force=force,
    )

    for experiment_name, experiment_param in zip(experiment_names, experiment_params):
        log.info("-%s :", experiment_name)
        configuration = copy.deepcopy(general_configuration)
        configuration.pop("models")
        for model_type, model_param in zip(model_types, model_params):
            log.info("--%s :", model_type)
            configuration["model_type"] = model_type
            configuration["model"] = model_param
            features_list_paths = copy.deepcopy(configuration["feature_paths"])
            for features_list_path in features_list_paths:
                input_configuration = copy.deepcopy(configuration)
                features_list_path_name = features_list_path
                log.info("---%s :", features_list_path_name)
                input_configuration["features"] = features_list_path

                tune_model = Tuning(
                    train_data=train_data,
                    test_data=test_data,
                    unlabeled_path=unlabeled_path,
                    configuration=input_configuration,
                    folder_name=folder_name,
                    experiment_name=experiment_name,
                    experiment_param=experiment_param,
                    sub_folder_name=features_list_path_name,
                    features_list_path=features_list_path_name,
                    features_configuration_path=train_data.features_configuration_path,
                    is_compute_metrics=False,
                )
                result = tune_model.train()
                results.append(result)
    results_df = pd.DataFrame(results)
    results_df.sort_values(["score"], inplace=True)
    best = results_df.iloc[0]
    log.info(
        "best run goes to the model %s using  %s " "and the features %s with  score %s ",
        best.model,
        best.experiment,
        best.features,
        best.score,
    )
    log.info(results_df.to_string(index=False))
    results_df.to_csv((experiment_path / "results.csv"), index=False)
    remove_unnecessary_folders(experiment_path, train_data.remove_unnecessary_folders)


def _train_func(  # noqa: CCR001
    experiment_name: str,
    experiment_param: Dict[str, Any],
    model_type: str,
    model_param: Dict[str, Any],
    train_data: Dataset,
    test_data: Dataset,
    unlabeled_path: str,
    configuration: Dict[str, Any],
    folder_name: str,
    log_handler: Logger,
    neptune_log: Optional[NeptuneLogs] = None,
    sub_folder_name: Optional[str] = None,
    comparison_score_metrics: Optional[pd.DataFrame] = None,
) -> Tuple[List[EvalExpType], str]:
    """Train the same experiment with different features lists."""
    display = ""

    experiment_configuration = _generate_single_exp_config(
        configuration=configuration,
        experiment_name=experiment_name,
        experiment_param=experiment_param,
        model_type=model_type,
        model_param=model_param,
    )

    experiment_class = import_experiment(exper, experiment_name)
    features_list_paths = copy.deepcopy(configuration["feature_paths"])
    eval_configuration = configuration["evaluation"]
    results = []
    display += f"\n -{model_type} : \n"
    for features_list_path in features_list_paths:
        sub_model_features = features_list_path
        log_handler.info("          -features set :%s", sub_model_features)
        features_sub_folder_name = (
            f"{sub_model_features}/{sub_folder_name}" if sub_folder_name else sub_model_features
        )

        experiment_configuration["features"] = features_list_path
        experiment = experiment_class(
            train_data=train_data,
            test_data=test_data,
            unlabeled_path=unlabeled_path,
            configuration=experiment_configuration,
            experiment_name=experiment_name,
            folder_name=folder_name,
            sub_folder_name=features_sub_folder_name,
            features_configuration_path=train_data.features_configuration_path,
            **experiment_param,
        )

        experiment.train()
        scores = experiment.eval_exp(comparison_score_metrics=comparison_score_metrics)

        validation_score = scores["validation"][eval_configuration["metric_selector"]].iloc[0]
        if not validation_score:
            validation_score = ""
        test_score = scores["test"][eval_configuration["metric_selector"]].iloc[0]
        prediction_name = scores["validation"]["prediction"].iloc[0]

        display += f"  * Features: {sub_model_features}\n"
        display += (
            f"     {prediction_name} :"
            f" Validation : {validation_score:0.3f}"
            f" Test:{test_score:0.3f}\n"
        )

        results.append(scores)
        if neptune_log:
            neptune_log.upload_experiment(experiment_path=experiment.experiment_directory)
    return results, display


def _check_model_folder(experiment_path: Path) -> None:
    """Check if the checkpoint folder  exists or not."""
    if experiment_path.exists():
        click.confirm(
            (
                f"The model folder with the name {experiment_path.name} already exists."
                "Do you want to continue the training but"
                "all the checkpoints will be deleted?"
            ),
            abort=True,
        )
        shutil.rmtree(experiment_path)

    experiment_path.mkdir(exist_ok=True, parents=True)


def remove_unnecessary_folders(experiment_path: Path, remove: bool) -> None:
    """Remove unnecessary folders inside the experiment directory."""
    subfolders_to_remove = EXP_NAMES + [DATAPROC_DIRECTORY]
    for directory in subfolders_to_remove:
        path = experiment_path / directory

        if remove and path.exists():

            if directory == DATAPROC_DIRECTORY:
                data_processing_files = path.iterdir()
                files_to_remove = list(
                    filter(
                        lambda x: x.suffix in [".csv", ".tsv", ".xlsx"], list(data_processing_files)
                    )
                )
                for file in files_to_remove:

                    file.unlink()
            else:
                shutil.rmtree(path)


def _generate_single_exp_config(
    configuration: Dict[str, Any],
    experiment_name: str,
    experiment_param: Dict[str, Any],
    model_type: str,
    model_param: Dict[str, Any],
) -> Dict[str, Union[str, Dict[str, str]]]:
    """Cleanup the configuration file for each Experiment."""
    configuration = copy.deepcopy(configuration)
    configuration.pop("feature_paths", None)
    configuration["processing"].pop("folds", None)
    configuration["processing"].pop("seeds", None)
    if experiment_name == KFOLD_MULTISEED_MODEL_NAME:
        if not experiment_param.get("seeds", None):
            experiment_param["seeds"] = generate_random_seeds(
                nbr_seeds=experiment_param["nbr_seeds"]
            )
    configuration["experiments"] = {experiment_name: experiment_param}
    configuration["model_type"] = model_type
    configuration["model"] = model_param

    return configuration


def _save_best_experiment(best_experiment_id: List[str], experiment_path: Path) -> None:
    """Save best experiment."""
    if len(best_experiment_id) == 4:
        ## TODO change the saving folder order to match 0,1,2,3 instead of 0,1,3,2
        path = (
            experiment_path
            / best_experiment_id[0]
            / best_experiment_id[1]
            / best_experiment_id[3]
            / best_experiment_id[2]
        )
    else:
        path = (
            experiment_path / best_experiment_id[0] / best_experiment_id[1] / best_experiment_id[2]
        )
    best_experiment_path = path
    destination_path = experiment_path / "best_experiment"
    shutil.copytree(src=best_experiment_path, dst=destination_path, dirs_exist_ok=True)


def load_datasets(
    click_ctx: Union[click.core.Context, Any],
    train_data_path: str,
    test_data_path: str,
    general_configuration: Dict[str, Any],
    experiment_path: Path,
    force: bool,
) -> Tuple[Dataset, Dataset]:
    """Load train , test data for a given paths."""
    train_data = Dataset(
        click_ctx=click_ctx,
        data_path=train_data_path,
        configuration=general_configuration,
        is_train=True,
        experiment_path=experiment_path,
        force_gcp=force,
    ).load_data()
    test_data = Dataset(
        click_ctx=click_ctx,
        data_path=test_data_path,
        configuration=general_configuration,
        is_train=False,
        experiment_path=experiment_path,
        force_gcp=force,
        is_inference=True,
        process_label=True,
    ).load_data()
    return train_data, test_data


@click.command()
@click.option(
    "--data_path",
    "-d",
    type=str,
    required=True,
    help="Path to the dataset.",
)
@click.option(
    "--label_name",
    "-l",
    type=str,
    required=True,
    default="cd8_any",
    help="label name.",
)
@click.option(
    "--column_name",
    "-c",
    type=str,
    required=True,
    help="label name.",
)
def compute_comparison_score(
    data_path: Union[str, Path], label_name: str, column_name: str
) -> None:
    """Compute The comparison scores for given column name."""
    data_path = Path(data_path)
    data = pd.read_csv(data_path)
    log.info("Evaluating %s in %s", column_name, data_path.name)
    if data[column_name].isna().sum():
        ratio, topk_ = topk_global(data[label_name], data[column_name])
        log.info("Topk : %s where %s positive label is captured ", ratio, topk_)

        log.info(
            (
                "%s contains missing values, %s",
                "rows will be dropped",
                column_name,
                data[column_name].isna().sum(),
            )
        )
        data = data[~data[column_name].isna()]
    if len(data):
        ratio, topk_ = topk_global(data[label_name], data[column_name])
        log.info("The new Topk")
        log.info("Topk : %s where %s positive label is captured ", ratio, topk_)
    else:
        log.info("data is empty check %s column.", column_name)


def eval_comparison_score(
    configuration: Dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    dataset_name: str,
    experiment_path: Path,
    plot_comparison_score_only: bool = True,
) -> Optional[pd.DataFrame]:  # noqa
    """Eval comparison score."""
    comparison_score = configuration["evaluation"].get("comparison_score", None)
    if comparison_score:
        log.info("Eval %s", comparison_score)
        results: Dict[str, MetricsEvalType] = {}

        experiment_names, _ = load_experiments(configuration)
        if SINGLE_MODEL_NAME in experiment_names:
            validation_column = configuration["experiments"][SINGLE_MODEL_NAME]["validation_column"]
            if validation_column in train_data.columns:

                curve_plot_directory = experiment_path / "comparison_score" / SINGLE_MODEL_NAME
                curve_plot_directory.mkdir(exist_ok=True, parents=True)
                evaluator = Evaluation(
                    label_name=configuration["label"],
                    eval_configuration=configuration["evaluation"],
                    curve_plot_directory=curve_plot_directory,
                    plot_comparison_score_only=plot_comparison_score_only,
                )

                log.info(SINGLE_MODEL_NAME)

                evaluator.compute_metrics(
                    data=train_data[train_data[validation_column] == 0],
                    prediction_name=comparison_score,
                    split_name="train",
                    dataset_name=dataset_name,
                )
                evaluator.compute_metrics(
                    data=train_data[train_data[validation_column] == 1],
                    prediction_name=comparison_score,
                    split_name="validation",
                    dataset_name=dataset_name,
                )
                results[SINGLE_MODEL_NAME] = evaluator.get_evals()
        kfold_exps = list(set(experiment_names) & set(KFOLD_EXP_NAMES))
        if kfold_exps:
            split_column = configuration["experiments"][kfold_exps[0]]["split_column"]
            if split_column in train_data.columns:

                log.info(KFOLD_MODEL_NAME)
                curve_plot_directory = experiment_path / "comparison_score" / KFOLD_MODEL_NAME
                curve_plot_directory.mkdir(exist_ok=True, parents=True)
                evaluator = Evaluation(
                    label_name=configuration["label"],
                    eval_configuration=configuration["evaluation"],
                    curve_plot_directory=curve_plot_directory,
                    plot_comparison_score_only=plot_comparison_score_only,
                )
                for split in np.sort(train_data[split_column].unique()):
                    log.info(split)
                    evaluator.compute_metrics(
                        data=train_data[train_data[split_column] != split],
                        prediction_name=comparison_score,
                        split_name=f"train_{split}",
                        dataset_name=dataset_name,
                    )
                    evaluator.compute_metrics(
                        data=train_data[train_data[split_column] == split],
                        prediction_name=comparison_score,
                        split_name=f"validation_{split}",
                        dataset_name=dataset_name,
                    )
                results[KFOLD_MODEL_NAME] = evaluator.get_evals()
        log.info("Test")
        curve_plot_directory = experiment_path / "comparison_score"
        evaluator = Evaluation(
            label_name=configuration["label"],
            eval_configuration=configuration["evaluation"],
            curve_plot_directory=curve_plot_directory,
            plot_comparison_score_only=plot_comparison_score_only,
        )
        evaluator.compute_metrics(
            data=test_data,
            prediction_name=comparison_score,
            split_name="test",
            dataset_name=dataset_name,
        )
        results["test"] = evaluator.get_evals()
        save_yml(
            results,
            experiment_path / "comparison_score" / f"eval_{comparison_score}.yml",
        )

        return parse_comparison_score_metrics_to_df(results, comparison_score)
    return None


def parse_comparison_score_metrics_to_df(
    metrics: Dict[str, MetricsEvalType], comparison_score: str
) -> pd.DataFrame:
    """Parse comparison score metrics to dataframe object."""
    results = []
    for key, values in metrics.items():
        for prediction, scores in values.items():
            result = {}
            result["experiments"] = key
            splites_pred = prediction.split("_")
            result["prediction"] = (
                "prediction" if len(splites_pred) == 1 else f"prediction_{splites_pred[-1]}"
            )
            result["split"] = prediction if len(splites_pred) == 0 else splites_pred[0]
            result["topk"] = scores[comparison_score]["global"]["topk"]
            result["roc"] = scores[comparison_score]["global"]["roc"]
            results.append(result)

    results_cs = pd.DataFrame(results)
    results_cs["type"] = comparison_score
    return results_cs
