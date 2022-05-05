"""Module used to define the training command."""
import copy
import shutil

import click
import numpy as np
import pandas as pd

import biondeep_ig.src.experiments as exper
from biondeep_ig import CONFIGURATION_DIRECTORY
from biondeep_ig import DATAPROC_DIRACTORY
from biondeep_ig import DEFAULT_SEED
from biondeep_ig import KFOLD_MODEL_NAME
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig import SINGLE_MODEL_NAME
from biondeep_ig.feature_selection import feature_selection_main
from biondeep_ig.src.evaluation import Evaluation
from biondeep_ig.src.experiments.tuning import Tuning
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.logger import NeptuneLogs
from biondeep_ig.src.processing_v1 import Dataset
from biondeep_ig.src.utils import copy_existing_featrues_lists
from biondeep_ig.src.utils import get_best_experiment
from biondeep_ig.src.utils import import_experiment
from biondeep_ig.src.utils import load_experiments
from biondeep_ig.src.utils import load_models
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import log_summary_results
from biondeep_ig.src.utils import save_yml
from biondeep_ig.src.utils import seed_basic

log = get_logger("Train")
seed_basic(DEFAULT_SEED)


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
def train(train_data_path, test_data_path, unlabeled_path, configuration_file, folder_name):
    """Launch the training of a model based on the provided configuration and the input training file."""
    experiment_path = MODELS_DIRECTORY / folder_name
    _check_model_folder(experiment_path)
    init_logger(folder_name)
    log.info("Started")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    eval_configuration = general_configuration["evaluation"]
    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(general_configuration)
    log.info("****************************** Load EXP ****************************** ")
    model_types, model_params = load_models(general_configuration)
    log.info("****************************** Load Models ****************************** ")

    train_data, test_data = load_datasets(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        general_configuration=general_configuration,
        experiment_path=experiment_path,
    )
    log.info("*********************** Load and process Train , Test *********************** ")
    existing_featrues_lists = general_configuration.get("feature_paths", [])
    if "FS" in general_configuration:

        features_names = feature_selection_main(
            train_data=train_data,
            configuration_file=configuration_file,
            folder_name=folder_name,
            with_train=True,
        )
        log.info(f"features_names : {' '.join(features_names)}")
        log.info("****************************** Finished FS ****************************** ")

        feature_paths = existing_featrues_lists + features_names
        general_configuration["feature_paths"] = feature_paths

    copy_existing_featrues_lists(
        existing_featrues_lists,
        experiment_path=experiment_path,
        label_name=general_configuration["label"],
    )
    displays = "####### Runs Summary #######"
    results = []
    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["simple train"], training_path=train_data_path, test_path=test_data_path
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)
    save_yml(general_configuration, experiment_path / "configuration.yml")

    comparison_score_metrics = eval_comparison_score(
        configuration=general_configuration,
        train_data=train_data().copy(),
        test_data=test_data().copy(),
        experiment_path=experiment_path,
    )
    for experiment_name, experiment_param in zip(experiment_names, experiment_params):
        log.info(f"{experiment_name} :")
        configuration = copy.deepcopy(general_configuration)
        configuration.pop("models")
        displays += f"\n Method :{experiment_name}: "
        for model_type, model_param in zip(model_types, model_params):
            log.info(f"  {model_type}:")
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
    best_experiment_id = best_experiment_id.split("//")
    neptune_log.upload_experiment(
        experiment_path=(
            experiment_path / best_experiment_id[0] / best_experiment_id[1] / best_experiment_id[2]
        ),
        neptune_sub_folder="best_experiment",
    )
    _save_best_experiment(best_experiment_id=best_experiment_id, experiment_path=experiment_path)
    remove_processed_data(experiment_path, train_data.remove_proc_data)


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
def train_seed_fold(  # noqa
    train_data_path, test_data_path, unlabeled_path, configuration_file, folder_name
):
    """Train with multiple seed and folds."""
    experiment_path = MODELS_DIRECTORY / folder_name
    _check_model_folder(experiment_path)
    init_logger(folder_name)
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    copy_existing_featrues_lists(
        general_configuration.get("feature_paths", []),
        experiment_path=experiment_path,
        label_name=general_configuration["label"],
    )
    experiment_names, experiment_params = load_experiments(general_configuration)
    model_types, model_params = load_models(general_configuration)
    seeds = general_configuration["processing"]["seeds"]
    folds = general_configuration["processing"]["folds"]
    eval_configuration = general_configuration["evaluation"]

    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["Multi seed fold train"], training_path=train_data_path, test_path=test_data_path
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)
    save_yml(general_configuration, experiment_path / "configuration.yml")
    log.info("****************************** Load YAML ****************************** ")
    log.info("****************************** Load EXP ****************************** ")
    log.info("****************************** Load Models ****************************** ")
    displays = "####### Runs Summary #######"
    results = []
    for experiment_name, experiment_param in zip(experiment_names, experiment_params):
        log.info(f"{experiment_name} :")
        configuration = copy.deepcopy(general_configuration)
        configuration.pop("models")
        displays += f"\n Method :{experiment_name}: "
        for model_type, model_param in zip(model_types, model_params):
            log.info(f"  -{model_type} :")
            for seed in seeds:
                log.info(f"      -seed {seed} :")
                configuration["processing"]["seed"] = seed
                if experiment_name == "SingleModel":
                    train_data, test_data = load_datasets(
                        train_data_path=train_data_path,
                        test_data_path=test_data_path,
                        general_configuration=configuration,
                        experiment_path=experiment_path,
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
                        e["test"]["ID"] = e["test"]["ID"] + f"//seed_{seed}"
                        e["validation"]["ID"] = e["validation"]["ID"] + f"//seed_{seed}"
                    results.extend(result)

                    displays += f"\n -Seed : {seed}" + display
                else:
                    for fold in folds:
                        log.info(f"         -fold : {fold} :")
                        configuration["processing"]["fold"] = fold
                        train_data, test_data = load_datasets(
                            train_data_path=train_data_path,
                            test_data_path=test_data_path,
                            general_configuration=configuration,
                            experiment_path=experiment_path,
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
    best_experiment_id = best_experiment_id.split("//")
    neptune_log.upload_experiment(
        experiment_path=(
            experiment_path
            / best_experiment_id[0]
            / best_experiment_id[1]
            / best_experiment_id[3]
            / best_experiment_id[2]
        ),
        neptune_sub_folder="best_experiment",
    )
    _save_best_experiment(best_experiment_id=best_experiment_id, experiment_path=experiment_path)
    remove_processed_data(experiment_path, train_data.remove_proc_data)


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
def tune(train_data_path, test_data_path, unlabeled_path, configuration_file, folder_name):
    """Tune model params."""
    experiment_path = MODELS_DIRECTORY / folder_name
    _check_model_folder(experiment_path)
    init_logger(folder_name)
    log.info("Started")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    copy_existing_featrues_lists(
        general_configuration.get("feature_paths", []),
        experiment_path=experiment_path,
        label_name=general_configuration["label"],
    )
    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(general_configuration)
    log.info("****************************** Load EXP ****************************** ")
    model_types, model_params = load_models(general_configuration)
    log.info("****************************** Load Models ****************************** ")
    results = []
    save_yml(general_configuration, experiment_path / "configuration.yml")
    train_data, test_data = load_datasets(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        general_configuration=general_configuration,
        experiment_path=experiment_path,
    )

    for experiment_name, experiment_param in zip(experiment_names, experiment_params):
        log.info(f"-{experiment_name} :")
        configuration = copy.deepcopy(general_configuration)
        configuration.pop("models")
        for model_type, model_param in zip(model_types, model_params):
            log.info(f"--{model_type} :")
            configuration["model_type"] = model_type
            configuration["model"] = model_param
            features_list_paths = copy.deepcopy(configuration["feature_paths"])
            for features_list_path in features_list_paths:
                input_cofiguration = copy.deepcopy(configuration)
                features_list_path_name = features_list_path
                log.info(f"---{features_list_path_name} :")
                input_cofiguration["features"] = features_list_path

                tune_model = Tuning(
                    train_data=train_data,
                    test_data=test_data,
                    unlabeled_path=unlabeled_path,
                    configuration=input_cofiguration,
                    folder_name=folder_name,
                    experiment_name=experiment_name,
                    experiment_param=experiment_param,
                    sub_folder_name=features_list_path_name,
                    is_compute_metrics=False,
                )
                result = tune_model.train(features_list_path_name)
                results.append(result)
    results = pd.DataFrame(results)
    results.sort_values(["score"], inplace=True)
    best = results.iloc[0]
    log.info(
        f"best run goes to the model {best.model} using  {best.experiment} "
        + f"and the features {best.features} with  score {best.score:.3f} "
    )
    log.info(results.to_string(index=False))
    results.to_csv((experiment_path / "results.csv"), index=False)
    remove_processed_data(experiment_path, train_data.remove_proc_data)


def _train_func(  # noqa: CCR001
    experiment_name,
    experiment_param,
    model_type,
    model_param,
    train_data,
    test_data,
    unlabeled_path,
    configuration,
    folder_name,
    log_handler,
    neptune_log=None,
    sub_folder_name=None,
    comparison_score_metrics=None,
):
    """Train the same experiment with different features lists."""
    display = ""

    experiment_configuration = _genrate_single_exp_config(
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
        log_handler.info(f"          -features set :{sub_model_features}")
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


def _check_model_folder(experiment_path):
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


def remove_processed_data(experiment_path, remove):
    """Remove data proc folder."""
    path = experiment_path / DATAPROC_DIRACTORY
    if remove:
        shutil.rmtree(path)


def _genrate_single_exp_config(
    configuration, experiment_name, experiment_param, model_type, model_param
):
    """Cleanup the configuration file for each Experiment."""
    configuration = copy.deepcopy(configuration)
    configuration.pop("feature_paths")
    configuration["processing"].pop("folds", None)
    configuration["processing"].pop("seeds", None)
    configuration["experiments"] = {experiment_name: experiment_param}
    configuration["model_type"] = model_type
    configuration["model"] = model_param
    return configuration


def _save_best_experiment(best_experiment_id, experiment_path):
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
    shutil.copytree(best_experiment_path, destination_path)


def load_datasets(train_data_path, test_data_path, general_configuration, experiment_path):
    """Load train , test data for a given paths."""
    train_data = Dataset(
        data_path=train_data_path,
        configuration=general_configuration,
        is_train=True,
        experiment_path=experiment_path,
    ).load_data()
    test_data = Dataset(
        data_path=test_data_path,
        configuration=general_configuration,
        is_train=False,
        experiment_path=experiment_path,
    ).load_data()
    return train_data, test_data


def eval_comparison_score(configuration, train_data, test_data, experiment_path):  # noqa
    """Eval comparation score."""
    comparison_score = configuration["evaluation"].get("comparison_score", None)
    if comparison_score:
        log.info(f"Eval {comparison_score}")
        results = {}

        test_data[comparison_score].fillna(test_data[comparison_score].mean(), inplace=True)
        train_data[comparison_score].fillna(train_data[comparison_score].mean(), inplace=True)

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
                )

                log.info(SINGLE_MODEL_NAME)

                evaluator.compute_metrics(
                    data=train_data[train_data[validation_column] == 0],
                    prediction_name=comparison_score,
                    data_name="train",
                )
                evaluator.compute_metrics(
                    data=train_data[train_data[validation_column] == 1],
                    prediction_name=comparison_score,
                    data_name="validation",
                )
                results[SINGLE_MODEL_NAME] = evaluator.get_evals()

        if KFOLD_MODEL_NAME in experiment_names:
            split_column = configuration["experiments"][KFOLD_MODEL_NAME]["split_column"]
            if split_column in train_data.columns:
                log.info(KFOLD_MODEL_NAME)
                curve_plot_directory = experiment_path / "comparison_score" / KFOLD_MODEL_NAME
                curve_plot_directory.mkdir(exist_ok=True, parents=True)
                evaluator = Evaluation(
                    label_name=configuration["label"],
                    eval_configuration=configuration["evaluation"],
                    curve_plot_directory=curve_plot_directory,
                )
                for split in np.sort(train_data[split_column].unique()):
                    log.info(split)
                    evaluator.compute_metrics(
                        data=train_data[train_data[split_column] != split],
                        prediction_name=comparison_score,
                        data_name=f"train_{split}",
                    )
                    evaluator.compute_metrics(
                        data=train_data[train_data[split_column] == split],
                        prediction_name=comparison_score,
                        data_name=f"validation_{split}",
                    )
                results[KFOLD_MODEL_NAME] = evaluator.get_evals()
        log.info("Test")
        curve_plot_directory = experiment_path / "comparison_score"
        evaluator = Evaluation(
            label_name=configuration["label"],
            eval_configuration=configuration["evaluation"],
            curve_plot_directory=curve_plot_directory,
        )
        evaluator.compute_metrics(
            data=test_data, prediction_name=comparison_score, data_name="test"
        )
        results["test"] = evaluator.get_evals()
        save_yml(
            results,
            experiment_path / "comparison_score" / f"eval_{comparison_score}.yml",
        )

        return parse_comparasion_score_metrics_to_df(results, comparison_score)
    return None


def parse_comparasion_score_metrics_to_df(metrics, comparison_score):
    """Parse comparasion score metrics to dataframe object."""
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
    results_cs["type"] = "Comparison score"
    return results_cs
