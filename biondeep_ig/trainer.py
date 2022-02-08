"""Module used to define the training command."""
import copy
import shutil

import click
import numpy as np
import pandas as pd

import biondeep_ig.src.experiments as exper
from biondeep_ig.feature_selection import feature_selection_main
from biondeep_ig.src import CONFIGURATION_DIRECTORY
from biondeep_ig.src import DATAPROC_DIRACTORY
from biondeep_ig.src import ID_NAME
from biondeep_ig.src import KFOLD_MODEL_NAME
from biondeep_ig.src import MODELS_DIRECTORY
from biondeep_ig.src import SINGLE_MODEL_NAME
from biondeep_ig.src.evaluation import Evaluation
from biondeep_ig.src.experiments.tuning import Tuning
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.logger import NeptuneLogs
from biondeep_ig.src.utils import get_best_experiment
from biondeep_ig.src.utils import get_model_module_by_name
from biondeep_ig.src.utils import load_experiments
from biondeep_ig.src.utils import load_models
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import log_summary_results
from biondeep_ig.src.utils import remove_genrated_features
from biondeep_ig.src.utils import save_yml

log = get_logger("Train")


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
    _check_model_folder(folder_name)
    init_logger(folder_name)
    log.info("Started")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    eval_configuration = general_configuration["evaluation"]
    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(general_configuration)
    log.info("****************************** Load EXP ****************************** ")
    model_types, model_params = load_models(general_configuration)
    log.info("****************************** Load Models ****************************** ")
    model_selection = get_safely_model_selection(general_configuration=general_configuration)
    log.info(f"Best model will be selected based on {model_selection}")

    if "FS" in general_configuration:
        features_names = feature_selection_main(
            train_data_path, test_data_path, configuration_file, folder_name
        )
        print("features_names", features_names)
        log.info("****************************** Finished FS ****************************** ")

        existing_featrues_lists = general_configuration.get("feature_paths", [])
        print("existing_featrues_lists", existing_featrues_lists)
        existing_featrues_lists.extend(features_names)
        general_configuration["feature_paths"] = existing_featrues_lists
    displays = "####### Runs Summary #######"
    results = []
    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["simple train"], training_path=train_data_path, test_path=test_data_path
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)
    save_yml(general_configuration, MODELS_DIRECTORY / folder_name / "configuration.yml")

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
                train_data_path=train_data_path,
                test_data_path=test_data_path,
                unlabeled_path=unlabeled_path,
                configuration=configuration,
                folder_name=folder_name,
                log_handler=log,
                neptune_log=neptune_log,
            )
            results.extend(result)
            displays += display
    log_summary_results(displays)
    best_experiment_id, eval_message = get_best_experiment(
        results, eval_configuration, path=MODELS_DIRECTORY / folder_name, file_name="results"
    )
    log_summary_results(eval_message)
    best_experiment_id = best_experiment_id.split("//")
    neptune_log.upload_experiment(
        experiment_path=(
            MODELS_DIRECTORY
            / folder_name
            / best_experiment_id[0]
            / best_experiment_id[1]
            / best_experiment_id[2]
        ),
        neptune_sub_folder="best_experiment",
    )
    _save_best_experiment(best_experiment_id=best_experiment_id, folder_name=folder_name)
    eval_comparison_score(
        configuration=general_configuration,
        train_path=train_data_path,
        test_path=test_data_path,
        folder_name=folder_name,
    )
    remove_processed_data(folder_name)

    if "FS" in general_configuration:
        remove_genrated_features(features_names, general_configuration["label"])


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
    _check_model_folder(folder_name)
    init_logger(folder_name)
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
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
    save_yml(general_configuration, MODELS_DIRECTORY / folder_name / "configuration.yml")
    model_selection = get_safely_model_selection(general_configuration=general_configuration)
    log.info("****************************** Load YAML ****************************** ")
    log.info("****************************** Load EXP ****************************** ")
    log.info("****************************** Load Models ****************************** ")
    log.info(f"Best model will be selected based on {model_selection}")
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
                    result, display = _train_func(
                        experiment_name=experiment_name,
                        experiment_param=experiment_param,
                        model_type=model_type,
                        model_param=model_param,
                        train_data_path=train_data_path,
                        test_data_path=test_data_path,
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
                        result, display = _train_func(
                            experiment_name=experiment_name,
                            experiment_param=experiment_param,
                            model_type=model_type,
                            model_param=model_param,
                            train_data_path=train_data_path,
                            test_data_path=test_data_path,
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

                            print(e["statistic"]["ID"])
                            print(e["validation"]["ID"])

                        results.extend(result)

                        displays += f"\n -Seed : {seed}  -Fold : {fold}" + display

    log_summary_results(displays)
    best_experiment_id, eval_message = get_best_experiment(
        results, eval_configuration, path=MODELS_DIRECTORY / folder_name, file_name="results"
    )
    log_summary_results(eval_message)
    best_experiment_id = best_experiment_id.split("//")
    neptune_log.upload_experiment(
        experiment_path=(
            MODELS_DIRECTORY
            / folder_name
            / best_experiment_id[0]
            / best_experiment_id[1]
            / best_experiment_id[3]
            / best_experiment_id[2]
        ),
        neptune_sub_folder="best_experiment",
    )
    _save_best_experiment(best_experiment_id=best_experiment_id, folder_name=folder_name)
    remove_processed_data(folder_name)


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
    _check_model_folder(folder_name)
    init_logger(folder_name)
    log.info("Started")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(general_configuration)
    log.info("****************************** Load EXP ****************************** ")
    model_types, model_params = load_models(general_configuration)
    log.info("****************************** Load Models ****************************** ")
    results = []
    save_yml(general_configuration, MODELS_DIRECTORY / folder_name / "configuration.yml")
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
                    train_data_path=train_data_path,
                    test_data_path=test_data_path,
                    unlabeled_path=unlabeled_path,
                    configuration=input_cofiguration,
                    folder_name=folder_name,
                    experiment_name=experiment_name,
                    experiment_param=experiment_param,
                    sub_folder_name=features_list_path_name,
                )
                result = tune_model.train(features_list_path_name)
                results.append(result)
    results = pd.DataFrame(results)
    results.sort_values(["score"], ascending=not tune_model.maximize, inplace=True)
    best = results.iloc[0]
    log.info(
        f"best run goes for the model {best.model} using the exp {best.experiment} "
        + f"and the features {best.features}  score {best.score} "
    )
    log.info(results)
    results.to_csv((MODELS_DIRECTORY / folder_name / "results.csv"), index=False)
    remove_processed_data(folder_name)


def _train_func(  # noqa: CCR001
    experiment_name,
    experiment_param,
    model_type,
    model_param,
    train_data_path,
    test_data_path,
    unlabeled_path,
    configuration,
    folder_name,
    log_handler,
    neptune_log=None,
    sub_folder_name=None,
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
    experiment_class = get_model_module_by_name(exper, experiment_name)
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
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            unlabeled_path=unlabeled_path,
            configuration=experiment_configuration,
            experiment_name=experiment_name,
            folder_name=folder_name,
            sub_folder_name=features_sub_folder_name,
            **experiment_param,
        )

        experiment.train()
        scores = experiment.eval_exp()

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


def _check_model_folder(folder_name):
    """Check if the checkpoint folder  exists or not."""
    model_folder_path = MODELS_DIRECTORY / folder_name
    if model_folder_path.exists():
        click.confirm(
            (
                f"The model folder with the name {folder_name} already exists."
                "Do you want to continue the training but"
                "all the checkpoints will be deleted?"
            ),
            abort=True,
        )
        shutil.rmtree(model_folder_path)

    model_folder_path.mkdir(exist_ok=True, parents=True)


def remove_processed_data(folder_name):
    """Remove data proc folder."""
    path = MODELS_DIRECTORY / folder_name / DATAPROC_DIRACTORY
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


# TODO remove model_selection and change it with prediction_name_selector
def get_safely_model_selection(general_configuration):
    """Load model selection column and check if it's included in the benchmark columns list or not."""
    model_selection = general_configuration.get("model_selection", "prediction_average")
    benchmark_column = general_configuration.get("benchmark_column", ["prediction_average"])
    if model_selection in benchmark_column:
        return model_selection
    raise KeyError(f"{model_selection}: is not available in the benchmark_column")


def _save_best_experiment(best_experiment_id, folder_name):
    """Save best experiment."""
    if len(best_experiment_id) == 4:
        ## TODO change the saving folder order to match 0,1,2,3 instead of 0,1,3,2
        path = (
            MODELS_DIRECTORY
            / folder_name
            / best_experiment_id[0]
            / best_experiment_id[1]
            / best_experiment_id[3]
            / best_experiment_id[2]
        )
    else:
        path = (
            MODELS_DIRECTORY
            / folder_name
            / best_experiment_id[0]
            / best_experiment_id[1]
            / best_experiment_id[2]
        )
    best_experiment_path = path
    destination_path = MODELS_DIRECTORY / folder_name / "best_experiment"
    shutil.copytree(best_experiment_path, destination_path)


def eval_comparison_score(configuration, train_path, test_path, folder_name):
    """Eval comparation score."""
    comparison_score = configuration["evaluation"].get("comparison_score", None)
    if comparison_score:
        log.info(f"Eval {comparison_score}")
        results = {}
        train_df = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test.columns = [col.lower() for col in test.columns]
        train_df.columns = [col.lower() for col in train_df.columns]
        train_df[ID_NAME] = range(len(train_df))

        train_df = train_df[
            (train_df[configuration["label"]] == 1) | (train_df[configuration["label"]] == 0)
        ]
        test = test[(test[configuration["label"]] == "1") | (test[configuration["label"]] == "0")]
        test[configuration["label"]] = test[configuration["label"]].astype(int)

        test[comparison_score].fillna(test[comparison_score].mean(), inplace=True)
        train_df[comparison_score].fillna(train_df[comparison_score].mean(), inplace=True)

        experiment_names, _ = load_experiments(configuration)
        if SINGLE_MODEL_NAME in experiment_names:
            curve_plot_directory = (
                MODELS_DIRECTORY / folder_name / "comparison_score" / SINGLE_MODEL_NAME
            )
            curve_plot_directory.mkdir(exist_ok=True, parents=True)
            evaluator = Evaluation(
                label_name=configuration["label"],
                eval_configuration=configuration["evaluation"],
                curve_plot_directory=curve_plot_directory,
            )
            split_validation_path = (
                MODELS_DIRECTORY
                / folder_name
                / "splits"
                / f"validation_split_{configuration['processing']['seed']}.csv"
            )
            if split_validation_path.exists():
                log.info(SINGLE_MODEL_NAME)
                validation_column = configuration["experiments"][SINGLE_MODEL_NAME][
                    "validation_column"
                ]
                split = pd.read_csv(split_validation_path)
                train_df = train_df.merge(split, on=[ID_NAME], how="left")
                evaluator.compute_metrics(
                    data=train_df[train_df[validation_column] == 0],
                    prediction_name=comparison_score,
                    data_name="train",
                )
                evaluator.compute_metrics(
                    data=train_df[train_df[validation_column] == 1],
                    prediction_name=comparison_score,
                    data_name="validation",
                )
                results[SINGLE_MODEL_NAME] = evaluator.get_evals()

        if KFOLD_MODEL_NAME in experiment_names:
            curve_plot_directory = (
                MODELS_DIRECTORY / folder_name / "comparison_score" / KFOLD_MODEL_NAME
            )
            curve_plot_directory.mkdir(exist_ok=True, parents=True)
            evaluator = Evaluation(
                label_name=configuration["label"],
                eval_configuration=configuration["evaluation"],
                curve_plot_directory=curve_plot_directory,
            )
            split_kfold_path = (
                MODELS_DIRECTORY
                / folder_name
                / "splits"
                / (
                    f"kfold_split_{configuration['processing']['fold']}"
                    f"_{configuration['processing']['seed']}"
                    ".csv"
                )
            )
            if split_kfold_path.exists():
                log.info(KFOLD_MODEL_NAME)
                split_column = configuration["experiments"][KFOLD_MODEL_NAME]["split_column"]
                split_kfold = pd.read_csv(split_kfold_path)
                train_df = train_df.merge(split_kfold, on=[ID_NAME], how="left")
                for split in np.sort(train_df[split_column].unique()):
                    log.info(split)
                    evaluator.compute_metrics(
                        data=train_df[train_df[split_column] != split],
                        prediction_name=comparison_score,
                        data_name=f"train_{split}",
                    )
                    evaluator.compute_metrics(
                        data=train_df[train_df[split_column] == split],
                        prediction_name=comparison_score,
                        data_name=f"validation_{split}",
                    )
                results[KFOLD_MODEL_NAME] = evaluator.get_evals()
        log.info("Test")
        curve_plot_directory = MODELS_DIRECTORY / folder_name / "comparison_score"
        evaluator = Evaluation(
            label_name=configuration["label"],
            eval_configuration=configuration["evaluation"],
            curve_plot_directory=curve_plot_directory,
        )
        evaluator.compute_metrics(data=test, prediction_name=comparison_score, data_name="test")
        results["test"] = evaluator.get_evals()
        save_yml(
            results,
            MODELS_DIRECTORY / folder_name / "comparison_score" / f"eval_{comparison_score}.yml",
        )
        return results
    return None
