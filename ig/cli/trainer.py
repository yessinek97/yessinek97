"""Module used to define the training command."""
import copy
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Union

import click
import pandas as pd

from ig import CONFIGURATION_DIRECTORY, DEFAULT_SEED, MODELS_DIRECTORY
from ig.bucket.click import arguments
from ig.cli.feature_selection import feature_selection_main
from ig.constants import EvalExpType, TuneResults
from ig.cross_validation.tuning import Tuning
from ig.utils.cross_validation import load_experiments, load_models
from ig.utils.embedding import save_probing_embeddings
from ig.utils.evaluation import eval_comparison_score
from ig.utils.general import get_features_paths, seed_basic
from ig.utils.io import check_model_folder, load_yml, save_yml
from ig.utils.logger import NeptuneLogs, get_logger, init_logger
from ig.utils.metrics import topk_global
from ig.utils.trainer import (
    copy_existing_features_lists,
    copy_models_configuration_files,
    get_best_experiment,
    load_datasets,
    log_summary_results,
    remove_unnecessary_folders,
    save_best_experiment,
    train_func,
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
        check_model_folder(experiment_path)
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

    # Save probing embeddings file in model directory
    if "LLMModel" in model_types:
        if (
            model_params[model_types.index("LLMModel")]["general_params"]["training_type"]
            == "probing"
        ):
            save_probing_embeddings(
                experiment_path,
                model_params[model_types.index("LLMModel")]["general_params"]["emb_file_path"],
            )

    log.info("*********************** Load and process Train , Test *********************** ")
    existing_features_lists: List[str] = get_features_paths(general_configuration)
    if "FS" in general_configuration:

        feature_paths = feature_selection_main(
            train_data=train_data,
            general_configuration=general_configuration,
            folder_name=folder_name,
            with_train=True,
        )
        log.info("features_names : %s", " ".join(feature_paths))
        log.info("****************************** Finished FS ****************************** ")
        if existing_features_lists:
            feature_paths = existing_features_lists + feature_paths

        general_configuration["feature_paths"] = feature_paths
    else:
        if len(existing_features_lists) == 0:
            raise ValueError(
                "Features Selection is not activated and features list are not provided, "
                "please check your configuration file"
            )

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
            result, display = train_func(
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
    save_best_experiment(
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
    check_model_folder(experiment_path)
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

    # Save probing embeddings file in model directory
    if "LLMModel" in model_types:
        if (
            model_params[model_types.index("LLMModel")]["general_params"]["training_type"]
            == "probing"
        ):
            save_probing_embeddings(
                experiment_path,
                model_params[model_types.index("LLMModel")]["general_params"]["emb_file_path"],
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
                    result, display = train_func(
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
                        result, display = train_func(
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
    save_best_experiment(
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
    seed_basic(DEFAULT_SEED)
    experiment_path = MODELS_DIRECTORY / folder_name
    check_model_folder(experiment_path)
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

    # Save probing embeddings file in model directory
    if "LLMModel" in model_types:
        if (
            model_params[model_types.index("LLMModel")]["general_params"]["training_type"]
            == "probing"
        ):
            save_probing_embeddings(
                experiment_path,
                model_params[model_types.index("LLMModel")]["general_params"]["emb_file_path"],
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
                    experiment_param=copy.deepcopy(experiment_param),
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
