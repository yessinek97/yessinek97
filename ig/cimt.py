"""Module used to define all the helper commands for cimt model."""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import ig.src.experiments as exper
from ig import CONFIGURATION_DIRECTORY, FEATURES_DIRECTORY, MODELS_DIRECTORY
from ig.src.dataset import Dataset
from ig.src.metrics import topk
from ig.src.utils import import_experiment, load_pkl, load_yml, read_data, save_yml
from ig.trainer import _check_model_folder, train


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
        print("Start splitting data to perform the Training per splits")
    else:
        print("Start splitting data to perform the features selection approach")
    data = read_data(data_path, low_memory=False)
    if configuration_path:
        configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    assert configuration is not None
    if is_train:
        seed = configuration["train"]["seed"]
        n_splits = configuration["train"]["n_splits"]
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
        print(f"split : {i+1} : Train : {len(train_split)},Test : {len(test_split)}")
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
    print("Start Features selection process.")
    if configuration_path:
        configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    assert configuration is not None
    train_configuration_file = configuration["features_selection"]["default_configuration"]
    train_configuration = load_yml(CONFIGURATION_DIRECTORY / train_configuration_file)
    train_configuration = update_configuration(
        train_configuration, configuration["features_selection"].get("configuration", {})
    )
    n_splits = configuration["features_selection"]["n_splits"]
    seed = configuration["features_selection"]["seed"]
    ntop_features = configuration["features_selection"]["Ntop_features"]
    train_fold = train_configuration["processing"]["fold"]
    if sub_folder:
        base_exp_path = MODELS_DIRECTORY / sub_folder / exp_name / "features_selection"
    else:
        base_exp_path = MODELS_DIRECTORY / exp_name / "features_selection"
    base_exp_path.mkdir(exist_ok=True, parents=True)
    save_yml(train_configuration, base_exp_path / "configuration.yml")

    for i in range(n_splits):
        train_data_path = Path(data_directory) / f"CIMTFeaturesSelectionTrain_{i}_{seed}.csv"
        test_data_path = Path(data_directory) / f"CIMTFeaturesSelectionTest_{i}_{seed}.csv"
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
    features_importance_extraction(base_exp_path, exp_name, n_splits, train_fold, ntop_features)


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
    print("Star the Training process")
    if configuration_path:
        configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    assert configuration is not None
    train_configuration_file = configuration["train"]["default_configuration"]
    train_configuration = load_yml(CONFIGURATION_DIRECTORY / train_configuration_file)
    train_configuration = update_configuration(
        train_configuration, configuration["train"].get("configuration", {})
    )
    n_splits = configuration["train"]["n_splits"]
    seed = configuration["train"]["seed"]
    label_name = train_configuration["label"]
    if sub_folder:
        base_exp_path = MODELS_DIRECTORY / sub_folder / exp_name / "train"
    else:
        base_exp_path = MODELS_DIRECTORY / exp_name / "train"
    base_exp_path.mkdir(exist_ok=True, parents=True)
    train_configuration["feature_paths"] = [f"{exp_name}_features"]
    save_yml(train_configuration, base_exp_path / "configuration.yml")

    for i in range(n_splits):
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
    get_train_average_score(
        base_exp_path,
        label_name,
        n_splits,
        train_configuration["evaluation"].get("comparison_score", None),
    )


def features_importance_extraction(
    base_exp_path: Path, exp_name: str, n_splits: int, train_fold: int, ntop_features: int
) -> None:
    """Extract features importance form different split and save the Ntop_fezatures."""
    features_importance_per_split = []
    for i in range(n_splits):
        split_exp_path = base_exp_path / f"split_{i}"
        features_importance_per_split.append(
            get_split_features_importance(split_exp_path, train_fold)
        )

    features_importance = pd.concat(features_importance_per_split)
    features_importance_mean = (
        features_importance.groupby("features")
        .value.mean()
        .rename("value")
        .reset_index()
        .sort_values("value")
    )
    features_importance_count = (
        features_importance.groupby("features").features.count().rename("count").reset_index()
    )
    features_importance = features_importance_mean.merge(
        features_importance_count, on="features", how="left"
    )
    features_importance["count"] /= n_splits
    features_importance["weight"] = features_importance.value * features_importance["count"]
    features_importance.sort_values("weight", inplace=True)
    list_of_important_features = features_importance[-ntop_features:].features.tolist()

    features_path = FEATURES_DIRECTORY / "CD8" / f"{exp_name}_features.txt"
    print(f"Save the Top {ntop_features} features to {features_path} ")
    save_features(list_of_important_features, features_path)
    features_importance.to_csv(base_exp_path.parent / "features_importance.csv", index=False)


def get_split_features_importance(base_exp_path: Path, train_fold: int) -> pd.DataFrame:
    """Get features importance per split."""
    importance_features_list = []
    for i in range(train_fold):
        path = base_exp_path / "best_experiment" / "checkpoint" / f"split_{i}" / "model.pkl"
        model = load_pkl(path)
        features_importance = model.model.get_score(importance_type="gain")
        importance_features_list.append(
            pd.DataFrame.from_dict(
                {"features": features_importance.keys(), "value": features_importance.values()}
            )
        )
        # save per split ##################################
    importance_features = pd.concat(importance_features_list)
    importance_features = (
        importance_features.groupby("features").value.mean().rename("value").reset_index()
    )
    return importance_features


def save_features(features: List[str], file_path: Path) -> None:
    """Save a list of features as a text format."""
    with open(file_path, "w") as f:
        for e in features:
            f.write(str(e) + "\n")


def get_train_average_score(
    base_exp_path: Path, label_name: str, n_splits: int, comparison_score: str
) -> None:
    """Extract and merge the prediction and evaluation per split and report average results."""
    splits_eval = []
    splits_pred = []
    comparison_score_eval = []
    for split in range(n_splits):
        series_model = pd.Series([], dtype=pd.StringDtype())
        exp_path = base_exp_path / f"split_{split}"

        split_eval_df = pd.read_csv(exp_path / "best_experiment" / "eval" / "results.csv")
        split_prediction_df = pd.read_csv(exp_path / "best_experiment" / "prediction" / "test.csv")
        split_eval_df_pred_mean = split_eval_df[split_eval_df.prediction == "prediction_mean"]
        series_model["name"] = f"split_{split}"
        series_model["Train"] = split_eval_df[split_eval_df.split == "train"].topk.mean()
        series_model["Validation"] = split_eval_df_pred_mean[
            split_eval_df_pred_mean.split == "validation"
        ].topk.iloc[0]
        series_model["Test"] = split_eval_df_pred_mean[
            split_eval_df_pred_mean.split == "test"
        ].topk.iloc[0]
        splits_eval.append(series_model)
        splits_pred.append(split_prediction_df)
        if comparison_score:
            comparison_score_file = exp_path / "comparison_score" / f"eval_{comparison_score}.yml"
            if comparison_score_file.exists():
                comparison_score_eval.append(
                    load_comparison_score_evaluation(
                        comparison_score_file, f"split_{split}", comparison_score
                    )
                )
    eval_df = pd.DataFrame(splits_eval)
    predictions = pd.concat(splits_pred)
    test_global_topk = topk(predictions[label_name], predictions.prediction_mean)
    average_train_topk = eval_df.Train.mean()
    average_validation_topk = eval_df.Validation.mean()
    average_test_topk = eval_df.Test.mean()

    print("Average Splits topk")
    print(f" -Train : {average_train_topk:.3}")
    print(f" -Validation : {average_validation_topk:.3}")
    print(f" -Test : {average_test_topk:.3}")
    print("Global topk")
    print(f" -Test : {test_global_topk:.3}")
    eval_df.to_csv(base_exp_path.parent / "eval_per_split.csv", index=False)
    predictions.to_csv(base_exp_path.parent / "predictions.csv", index=False)
    results = {
        "Average": {
            "Train": float(average_train_topk),
            "Validation": float(average_validation_topk),
            "Test": float(average_test_topk),
        },
        "Global": {"Test": float(test_global_topk)},
    }
    if comparison_score_eval:
        comparison_score_eval_df = pd.DataFrame(comparison_score_eval)
        average_comparison_score_test_topk = comparison_score_eval_df.Test.mean()
        print(f"Eval {comparison_score}")
        print("Average Splits topk")
        print(f" -Test : {average_comparison_score_test_topk:.3}")
        cs_results = {
            "comparison score": {
                "Average": {
                    "Test": float(average_comparison_score_test_topk),
                },
            }
        }
        if comparison_score in predictions.columns:
            global_comparison_score_test_topk = topk(
                predictions[label_name], predictions[comparison_score]
            )
            print("Global topk")
            print(f" -Test : {global_comparison_score_test_topk:.3}")
            cs_results["comparison score"].update({"Global": {"Test": float(test_global_topk)}})
    save_yml(results, base_exp_path.parent / "scores.yml")


def load_comparison_score_evaluation(
    comparison_score_file: Path, split: str, comparison_score: str
) -> pd.Series:
    """Load and extract comparison score from comparison score evaluation."""
    eval_score = load_yml(comparison_score_file)
    series = pd.Series([], dtype=pd.StringDtype())
    series["split"] = split
    series["Test"] = eval_score["test"]["test"][comparison_score]["global"]["topk"]
    return series


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
@click.option("--exp_name", "-n", type=str, required=True, help="Experiment name.")
@click.pass_context
def cimt(
    ctx: Union[click.core.Context, Any], data_path: str, configuration_path: str, exp_name: str
) -> None:
    """Train and Apply features selection approach for a given data."""
    exp_path = MODELS_DIRECTORY / exp_name
    _check_model_folder(exp_path)
    configuration_directory = exp_path / "configuration"
    configuration_directory.mkdir(exist_ok=True, parents=True)
    data_directory = str(Path(data_path).parent)
    main_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    save_yml(main_configuration, configuration_directory / "configuration.yml")
    base_configuration = main_configuration["General"]
    experiments = main_configuration["experiments"]
    ctx.invoke(
        cimt_kfold_split,
        data_path=data_path,
        configuration_path=None,
        is_train=False,
        configuration=base_configuration,
    )
    ctx.invoke(
        cimt_kfold_split,
        data_path=data_path,
        configuration_path=None,
        is_train=True,
        configuration=base_configuration,
    )
    for name, exp_configuration in zip(experiments.keys(), experiments.values()):
        configuration = deepcopy(base_configuration)
        configuration["features_selection"]["configuration"] = exp_configuration.get(
            "features_selection", {}
        ).get("configuration", {})
        configuration["train"]["configuration"] = exp_configuration.get("train", {}).get(
            "configuration", {}
        )
        save_yml(configuration, configuration_directory / f"{name}.yml")

        ctx.invoke(
            cimt_features_selection,
            data_directory=data_directory,
            configuration_path=None,
            exp_name=name,
            configuration=configuration,
            sub_folder=exp_name,
        )

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
@click.option("--process", "-p", is_flag=True, help="process and clean the provided data set")
@click.pass_context
def cimt_inference(  # noqa
    ctx: Union[click.core.Context, Any],
    data_path: str,
    exp_name: str,
    is_eval: bool,
    process: bool,
    comparison_score: Optional[str],
) -> None:
    """Inferring Method.

    Args:
        ctx: Click context manager
        data_path: test path
        exp_name: checkpoint name
        comparison_score : comparison_score  column name
        is_eval: IF True eval the provided data set
        process: if True process and clean the provided data set
    """
    eval_dict: Dict[str, Any] = {}
    main_exp_path = MODELS_DIRECTORY / exp_name
    main_configuration = load_yml(main_exp_path / "configuration" / "configuration.yml")
    base_configuration = main_configuration["General"]
    experiments_name = main_configuration["experiments"].keys()
    train_n_splits = base_configuration["train"]["n_splits"]
    for experiment_name in experiments_name:
        experiment_path = main_exp_path / experiment_name / "train"
        split_predictions = []
        for split in range(train_n_splits):
            split_experiment_path = experiment_path / f"split_{split}"
            split_experiment, data_loader = load_split_experiment_dataloader(
                ctx, split_experiment_path, data_path, process
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
            print(experiment_name)
            print(" Splits eval")
            splits_topk = []
            split_eval = {}
            for i, df in enumerate(split_predictions):
                split_topk = topk(df[label_name], df[prediction_columns_name])
                print(f"  Eval split {i} : {split_topk:.4}")
                splits_topk.append(split_topk)
                split_eval[f"split_{i}"] = split_topk
            print(f"  Min Topk: {np.min(splits_topk)}")
            print(f"  Mean Topk: {np.mean(splits_topk)}")
            print(f"  Max Topk: {np.max(splits_topk)}\n")
            split_eval["Min"] = float(np.min(splits_topk))
            split_eval["Mean"] = float(np.mean(splits_topk))
            split_eval["Max"] = float(np.max(splits_topk))
            exp_eval["split"] = split_eval
            global_topk = topk(
                test_prediction[label_name], test_prediction[prediction_columns_name]
            )
            print(f" Global Topk : {global_topk}")
            exp_eval["Global"] = {"Test": global_topk}
            eval_dict[experiment_name] = exp_eval
            if comparison_score:
                if comparison_score in data_loader().columns:
                    comparison_score_topk = topk(
                        data_loader()[label_name], data_loader()[comparison_score]
                    )
                    eval_dict[comparison_score] = comparison_score_topk
                    print(f"Eval {comparison_score}: {comparison_score_topk}")
    if is_eval:
        save_yml(eval_dict, main_exp_path / f"eval_{Path(data_path).stem}.yml")


def load_split_experiment_dataloader(
    ctx: Union[click.core.Context, Any], split_experiment_path: Path, data_path: str, process: bool
) -> Tuple[Any, Dataset]:
    """Load Experiment and dataloader for the given split_experiment_path and data_path."""
    best_exp_path = split_experiment_path / "best_experiment"

    exp_configuration = load_yml(best_exp_path / "configuration.yml")
    experiment_name = list(exp_configuration["experiments"])[0]
    experiment_params = exp_configuration["experiments"][experiment_name]
    features_file_path = best_exp_path / "features.txt"
    data_loader = Dataset(
        click_ctx=ctx,
        data_path=data_path,
        configuration=exp_configuration,
        is_train=False,
        experiment_path=best_exp_path.parent,
        forced_processing=process,
        force_gcp=True,
        is_inference=True,
        process_label=False,
    ).load_data()
    experiment_class = import_experiment(exper, experiment_name)
    experiment = experiment_class(
        train_data=None,
        test_data=data_loader,
        configuration=exp_configuration,
        experiment_name=experiment_name,
        folder_name=None,
        sub_folder_name=None,
        experiment_directory=best_exp_path,
        features_file_path=features_file_path,
        **experiment_params,
    )
    return experiment, data_loader


def aggregate_predictions(
    data: pd.DataFrame,
    split_predictions: List[pd.DataFrame],
    ids: List[str],
    features: List[str],
    prediction_columns_name: str,
    label_name: str,
) -> pd.DataFrame:
    """Aggregate predictions in order to return average predictions of different splits."""
    label_pred = pd.concat(split_predictions)[ids + [prediction_columns_name]]
    label_pred = label_pred.groupby(ids)[[prediction_columns_name]].mean().reset_index()
    columns = ids + features + [label_name]
    data = data[[col for col in columns if col in data.columns]]
    test_data = data.merge(label_pred, on=ids, how="left")
    return test_data


def update_configuration(
    default_configuration: Dict[str, Any], experiment_configuration: Dict[str, Any]
) -> Dict[str, Any]:
    """Update the default configuration file with the given experiment configuration."""
    configuration = deepcopy(default_configuration)
    for key in experiment_configuration.keys():
        if isinstance(experiment_configuration[key], Dict):
            for second_key in experiment_configuration[key].keys():
                key_configuration = configuration[key]
                key_configuration[second_key] = experiment_configuration[key][second_key]
            configuration[key] = key_configuration
        else:
            configuration[key] = experiment_configuration[key]

    return configuration
