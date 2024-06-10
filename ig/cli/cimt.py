"""Module used to define all the commands for cimt model."""
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Union

import click
import numpy as np
from sklearn.model_selection import KFold

from ig import CONFIGURATION_DIRECTORY, DEFAULT_SEED, MODELS_DIRECTORY
from ig.cli.trainer import train
from ig.utils.cimt import (
    CsEvalType,
    aggregate_predictions,
    compute_comparison_scores_metrics_per_split,
    experiments_evaluation,
    features_importance_extraction,
    get_experiment_average_score,
    load_split_experiment_dataloader,
    update_configuration,
)
from ig.utils.general import seed_basic
from ig.utils.io import check_model_folder, load_yml, read_data, save_yml
from ig.utils.logger import get_logger, init_logger
from ig.utils.metrics import topk

log: Logger = get_logger("CIMT")
seed_basic(DEFAULT_SEED)


@click.command()
@click.option(
    "--data_path",
    "-d",
    type=str,
    required=True,
    help="Path to the dataset to be splitted",
)
@click.option(
    "--configuration_path",
    "-c",
    type=str,
    required=True,
    help="Path to the configuration file",
)
@click.option(
    "--is_train", "-t", is_flag=True, help="if kfodl split is for train or features selection."
)
def cimt_kfold_split(
    data_path: str,
    is_train: bool,
    configuration: Optional[Dict[str, Any]] = None,
    configuration_path: Optional[str] = None,
) -> None:
    """Split the given data to different split using Kfold."""
    if is_train:
        log.info("Start splitting data to perform the Training per splits")
    else:
        log.info("Start splitting data to perform the features selection approach")
    data = read_data(data_path, low_memory=False)
    if configuration_path:
        configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    assert configuration is not None
    if is_train:
        seed = configuration["w_train"]["seed"]
        n_splits = configuration["w_train"]["n_splits"]
        data_name = "CIMTTraining"
    else:
        seed = configuration["features_selection"]["seed"]
        n_splits = configuration["features_selection"]["n_splits"]
        data_name = "CIMTFeaturesSelection"
    kfold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits_folder = Path(data_path).parent

    for i, (_, indx) in enumerate(kfold.split(data.index)):
        test_split = data[data.index.isin(indx)]
        train_split = data[~data.index.isin(indx)]
        log.info(f"split : {i+1} : Train : {len(train_split)},Test : {len(test_split)}")
        test_split.to_csv(splits_folder / f"{data_name}Test_{i}_{seed}.csv", index=False)
        train_split.to_csv(splits_folder / f"./{data_name}Train_{i}_{seed}.csv", index=False)


@click.command()
@click.option(
    "--data_directory",
    "-d",
    type=str,
    required=True,
    help="folder which contains the splitted datasets.",
)
@click.option(
    "--configuration_path",
    "-c",
    type=str,
    required=True,
    help="Path to the configuration file",
)
@click.option("--exp_name", "-n", type=str, required=True, help="Experiment name.")
@click.pass_context
def cimt_features_selection(
    ctx: Union[click.core.Context, Any],
    data_directory: str,
    exp_name: str,
    configuration: Optional[Dict[str, Any]] = None,
    configuration_path: Optional[str] = None,
    sub_folder: Optional[str] = None,
) -> None:
    """Apply features selection approach per split and average importance features."""
    if configuration_path:
        configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    assert configuration is not None
    train_configuration_file = configuration["features_selection"]["default_configuration"]
    do_w_train = configuration.get("do_w_train", False)
    train_configuration = load_yml(CONFIGURATION_DIRECTORY / train_configuration_file)
    train_configuration = update_configuration(
        train_configuration, configuration["features_selection"].get("configuration", {})
    )
    n_splits = configuration["features_selection"]["n_splits"]
    seed = configuration["features_selection"]["seed"]
    ntop_features = configuration["features_selection"]["Ntop_features"]
    train_fold = train_configuration["processing"]["fold"]
    do_w_train = configuration.get("do_w_train", False)
    comparison_columns = configuration["eval"].get("comparison_columns", [])
    label_column = configuration["eval"]["label_column"]
    exp_dir = exp_name + "/features_selection" if do_w_train else exp_name

    if sub_folder:
        base_exp_path = MODELS_DIRECTORY / sub_folder / exp_dir
    else:
        base_exp_path = MODELS_DIRECTORY / exp_dir
    base_exp_path.mkdir(exist_ok=True, parents=True)
    save_yml(train_configuration, base_exp_path / "configuration.yml")

    cs_evaluations: Dict[str, CsEvalType] = {}
    sub_experiments_path = []
    for i in range(n_splits):
        log.info(f"  Train Split {i+1}/{n_splits}")

        if configuration.get("split_data", False):
            train_data_path = Path(data_directory) / f"CIMTFeaturesSelectionTrain_{i}_{seed}.csv"
            test_data_path = Path(data_directory) / f"CIMTFeaturesSelectionTest_{i}_{seed}.csv"
        else:
            train_data_path = Path(data_directory) / configuration["features_selection"][
                "train_file_pattern"
            ].format(i)
            test_data_path = Path(data_directory) / configuration["features_selection"][
                "test_file_pattern"
            ].format(i)
        ctx.invoke(
            train,
            train_data_path=str(train_data_path),
            test_data_path=str(test_data_path),
            unlabeled_path=None,
            configuration_file=base_exp_path / "configuration.yml",
            folder_name=base_exp_path / f"split_{i}",
            force=False,
            multi_train=True,
        )
        init_logger(
            logging_directory=base_exp_path.parent,
            file_name="cimt",
        )

        sub_experiments_path.append(base_exp_path / f"split_{i}")
        cs_evaluations[f"split_{i}"] = compute_comparison_scores_metrics_per_split(
            comparison_columns=comparison_columns,
            label=label_column,
            train_path=str(train_data_path),
            test_path=str(test_data_path),
        )
    if do_w_train:
        features_importance_extraction(
            base_exp_path, exp_name, n_splits, train_fold, ntop_features, do_w_train
        )

    experiment_results_path = base_exp_path / "results"
    experiment_results_path.mkdir(exist_ok=True, parents=True)
    experiments_evaluation(
        sub_experiments_path=sub_experiments_path,
        experiment_path=base_exp_path,
        eval_params=configuration["eval"],
        cs_evaluations=cs_evaluations,
        comparison_columns=comparison_columns,
        label=label_column,
    )


@click.command()
@click.option(
    "--data_directory",
    "-d",
    type=str,
    required=True,
    help="folder which contains the splitted datasets.",
)
@click.option(
    "--configuration_path",
    "-c",
    type=str,
    required=True,
    help="Path to the configuration file",
)
@click.option("--exp_name", "-n", type=str, required=True, help="Experiment name.")
@click.pass_context
def cimt_train(
    ctx: Union[click.core.Context, Any],
    data_directory: str,
    exp_name: str,
    configuration: Optional[Dict[str, Any]] = None,
    configuration_path: Optional[str] = None,
    sub_folder: Optional[str] = None,
) -> None:
    """Train per split and report the return split predictions."""
    if configuration_path:
        configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    assert configuration is not None
    train_configuration_file = configuration["w_train"]["default_configuration"]
    train_configuration = load_yml(CONFIGURATION_DIRECTORY / train_configuration_file)
    train_configuration = update_configuration(
        train_configuration, configuration["w_train"].get("configuration", {})
    )
    n_splits = configuration["w_train"]["n_splits"]
    seed = configuration["w_train"]["seed"]
    label_name = train_configuration["label"]
    if sub_folder:
        base_exp_path = MODELS_DIRECTORY / sub_folder / exp_name / "train"
    else:
        base_exp_path = MODELS_DIRECTORY / exp_name / "train"
    base_exp_path.mkdir(exist_ok=True, parents=True)
    train_configuration["feature_paths"] = [f"{exp_name}_features"]
    save_yml(train_configuration, base_exp_path / "configuration.yml")

    for i in range(n_splits):
        log.info(f"  Train Split {i+1}/{n_splits}")
        train_data_path = Path(data_directory) / f"CIMTTrainingTrain_{i}_{seed}.csv"
        test_data_path = Path(data_directory) / f"CIMTTrainingTest_{i}_{seed}.csv"
        ctx.invoke(
            train,
            train_data_path=str(train_data_path),
            test_data_path=str(test_data_path),
            unlabeled_path=None,
            configuration_file=base_exp_path / "configuration.yml",
            folder_name=base_exp_path / f"split_{i}",
            force=False,
            multi_train=True,
        )
        init_logger(
            logging_directory=base_exp_path,
            file_name="cimt",
        )
    get_experiment_average_score(
        base_exp_path,
        label_name,
        n_splits,
        train_configuration["evaluation"].get("comparison_score", None),
    )


@click.command()
@click.option(
    "--data_path",
    "-d",
    type=str,
    required=False,
    help="Path to the dataset to be splitted",
    default=None,
)
@click.option(
    "--data_directory",
    "-dr",
    type=str,
    required=False,
    help="directory to the splitted dataset",
    default=None,
)
@click.option(
    "--configuration_path",
    "-c",
    type=str,
    required=True,
    help="Path to the configuration file",
)
@click.option("--exp_name", "-n", type=str, required=True, help="Experiment name.")
@click.pass_context
def cimt(
    ctx: Union[click.core.Context, Any],
    data_path: Optional[str],
    data_directory: Optional[str],
    configuration_path: str,
    exp_name: str,
) -> None:
    """Train and Apply features selection approach for a given data."""
    exp_path = MODELS_DIRECTORY / exp_name
    check_model_folder(exp_path)
    init_logger(
        logging_directory=exp_path,
        file_name="cimt",
    )
    log.info("Start Multi Train loop for CIMT.")
    configuration_directory = exp_path / "configuration"
    configuration_directory.mkdir(exist_ok=True, parents=True)
    main_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    save_yml(main_configuration, configuration_directory / "configuration.yml")
    base_configuration = main_configuration["General"]
    experiments = main_configuration["experiments"]
    if base_configuration.get("split_data", False):
        if data_path is None:
            raise ValueError("data_path must be defined when split_data is True")

        data_directory = str(Path(data_path).parent)
        log.info("Split Train data using Kfold for features selection ")
        ctx.invoke(
            cimt_kfold_split,
            data_path=data_path,
            configuration_path=None,
            is_train=False,
            configuration=base_configuration,
        )
        if base_configuration.get("do_w_train", False):
            log.info("Split Train data using Kfold for Training")
            ctx.invoke(
                cimt_kfold_split,
                data_path=data_path,
                configuration_path=None,
                is_train=True,
                configuration=base_configuration,
            )
    if data_directory is None:
        raise ValueError("data_directory argument should be defined when split_data is False")

    for name, exp_configuration in zip(experiments.keys(), experiments.values()):
        configuration = deepcopy(base_configuration)
        if exp_configuration is None:
            exp_configuration = {}
        configuration["features_selection"]["configuration"] = exp_configuration.get(
            "features_selection", {}
        ).get("configuration", {})
        configuration["w_train"]["configuration"] = exp_configuration.get("w_train", {}).get(
            "configuration", {}
        )
        save_yml(configuration, configuration_directory / f"{name}.yml")
        log.info(f"Start features selection and Training for {name}")
        ctx.invoke(
            cimt_features_selection,
            data_directory=data_directory,
            configuration_path=None,
            exp_name=name,
            configuration=configuration,
            sub_folder=exp_name,
        )

        if base_configuration.get("do_w_train", False):
            log.info(f"Start weighted features training for {name}")
            ctx.invoke(
                cimt_train,
                data_directory=data_directory,
                configuration_path=None,
                exp_name=name,
                configuration=configuration,
                sub_folder=exp_name,
            )


@click.command()
@click.option(
    "--data_path",
    "-d",
    type=str,
    required=True,
    help="Path to the dataset to be splitted",
)
@click.option("--exp_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--comparison_score", "-cs", type=str, required=False, help="Experiment name.", default=None
)
@click.option("--is_eval", "-e", is_flag=True, help="eval predictions")
@click.pass_context
def cimt_inference(  # noqa
    ctx: Union[click.core.Context, Any],
    data_path: str,
    exp_name: str,
    is_eval: bool,
    comparison_score: Optional[str],
) -> None:
    """Inferring Method.

    Args:
        ctx: Click context manager
        data_path: test path
        exp_name: checkpoint name
        comparison_score : comparison_score  column name
        is_eval: IF True eval the provided data set
    """
    eval_dict: Dict[str, Any] = {}
    main_exp_path = MODELS_DIRECTORY / exp_name
    main_configuration = load_yml(main_exp_path / "configuration" / "configuration.yml")
    base_configuration = main_configuration["General"]
    experiments_name = main_configuration["experiments"].keys()
    train_n_splits = base_configuration["w_train"]["n_splits"]
    for experiment_name in experiments_name:
        experiment_path = main_exp_path / experiment_name / "train"
        split_predictions = []
        for split in range(train_n_splits):
            split_experiment_path = experiment_path / f"split_{split}"
            split_experiment, data_loader = load_split_experiment_dataloader(
                ctx, split_experiment_path, data_path
            )
            split_prediction = split_experiment.inference(data_loader(), save_df=False)
            split_prediction["split"] = f"split_{split}"
            split_predictions.append(split_prediction)

        prediction_columns_name = split_experiment.configuration["evaluation"][
            "prediction_name_selector"
        ]
        label_name = split_experiment.label_name

        test_prediction = aggregate_predictions(
            data_loader(),
            split_predictions,
            [data_loader.id_column],
            split_experiment.features,
            prediction_columns_name,
            label_name,
        )
        test_prediction.to_csv(experiment_path.parent / (Path(data_path).stem + ".csv"))
        if is_eval:
            exp_eval = {}
            assert label_name in test_prediction.columns
            log.info(experiment_name)
            log.info(" Splits eval")
            splits_topk = []
            split_eval = {}
            for i, df in enumerate(split_predictions):
                split_topk = topk(df[label_name], df[prediction_columns_name])
                log.info(f"  Eval split {i} : {split_topk:.4}")
                splits_topk.append(split_topk)
                split_eval[f"split_{i}"] = split_topk
            log.info(f"  Min Topk: {np.min(splits_topk)}")
            log.info(f"  Mean Topk: {np.mean(splits_topk)}")
            log.info(f"  Max Topk: {np.max(splits_topk)}\n")
            split_eval["Min"] = float(np.min(splits_topk))
            split_eval["Mean"] = float(np.mean(splits_topk))
            split_eval["Max"] = float(np.max(splits_topk))
            exp_eval["split"] = split_eval
            global_topk = topk(
                test_prediction[label_name], test_prediction[prediction_columns_name]
            )
            log.info(f" Global Topk : {global_topk}")
            exp_eval["Global"] = {"Test": global_topk}
            eval_dict[experiment_name] = exp_eval
            if comparison_score:
                if comparison_score in data_loader().columns:
                    comparison_score_topk = topk(
                        data_loader()[label_name], data_loader()[comparison_score]
                    )
                    eval_dict[comparison_score] = comparison_score_topk
                    log.info(f"Eval {comparison_score}: {comparison_score_topk}")
    if is_eval:
        save_yml(eval_dict, main_exp_path / f"eval_{Path(data_path).stem}.yml")
