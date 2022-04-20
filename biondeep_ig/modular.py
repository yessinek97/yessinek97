"""Module used to define the modular model."""
import shutil

import click
import pandas as pd
from sklearn.linear_model import LinearRegression

from biondeep_ig import CONFIGURATION_DIRECTORY
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig import SINGLE_MODEL_NAME
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.logger import NeptuneLogs
from biondeep_ig.src.processing import Datasetold
from biondeep_ig.src.utils import load_models
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import save_yml
from biondeep_ig.trainer import _check_model_folder
from biondeep_ig.trainer import train

log = get_logger("Modular")


@click.command()  # noqa: CCR001
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
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
@click.pass_context
def modulartrain(ctx, train_data_path, test_data_path, configuration_file, folder_name):
    """Modular training."""
    _check_model_folder(folder_name)
    init_logger(folder_name)
    log.info("Started")
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)

    neptune_log = NeptuneLogs(general_configuration, folder_name=folder_name)
    neptune_log.init_neptune(
        tags=["modular train"], training_path=train_data_path, test_path=test_data_path
    )
    neptune_log.upload_configuration_files(CONFIGURATION_DIRECTORY / configuration_file)
    save_yml(general_configuration, MODELS_DIRECTORY / folder_name / "configuration.yml")
    # TODO remove benchmark_column
    benchmark_column = general_configuration.get("benchmark_column", ["prediction_average"])
    temp_directory = MODELS_DIRECTORY / folder_name / "temp"
    temp_directory.mkdir(exist_ok=True, parents=True)
    predictions_df_train = []
    predictions_df_test = []
    feature_l = []
    for i_conf, config_file in enumerate(general_configuration["configs"]):
        log.info(f"{config_file} :")
        local_config = load_yml(CONFIGURATION_DIRECTORY / config_file)
        _check_local_config(
            local_config, SINGLE_MODEL_NAME, general_configuration, benchmark_column
        )
        ctx.invoke(
            train,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            configuration_file=config_file,
            folder_name=folder_name + "/" + config_file.split(".")[0],
        )
        cur_modeltype = load_models(local_config)[0][0]
        predictions_df_train.append(
            pd.read_csv(
                f"{MODELS_DIRECTORY}/{folder_name}/{config_file.split('.')[0]}/{SINGLE_MODEL_NAME}"
                + f"/{local_config['feature_paths'][0]}/{cur_modeltype}/prediction/train.csv"
            )
        )
        predictions_df_test.append(
            pd.read_csv(
                f"{MODELS_DIRECTORY}/{folder_name}/{config_file.split('.')[0]}/{SINGLE_MODEL_NAME}"
                + "/{local_config['feature_paths'][0]}/{cur_modeltype}/prediction/test.csv"
            )
        )

        if i_conf == 0:
            df_train = pd.DataFrame(
                predictions_df_train[i_conf]["prediction"].rename(f"Feature_Contraction_{i_conf}")
            )
            df_train[general_configuration["label"].lower()] = predictions_df_train[i_conf][
                general_configuration["label"].lower()
            ]
            df_test = pd.DataFrame(
                predictions_df_test[i_conf]["prediction"].rename(f"Feature_Contraction_{i_conf}")
            )
            df_test[general_configuration["label"].lower()] = predictions_df_test[i_conf][
                general_configuration["label"].lower()
            ]
            if general_configuration.get("comparison_score", None):
                df_train[general_configuration["comparison_score"].lower()] = predictions_df_train[
                    i_conf
                ][general_configuration["comparison_score"].lower()]
                df_test[general_configuration["comparison_score"].lower()] = predictions_df_test[
                    i_conf
                ][general_configuration["comparison_score"].lower()]
                feature_l.append(general_configuration["comparison_score"].lower())
            feature_l.append(f"Feature_Contraction_{i_conf}".lower())
            feature_l.append(general_configuration["label"].lower())
        else:
            df_train[f"Feature_Contraction_{i_conf}"] = predictions_df_train[i_conf]["prediction"]
            df_test[f"Feature_Contraction_{i_conf}"] = predictions_df_test[i_conf]["prediction"]

        feature_l.append(f"Feature_Contraction_{i_conf}".lower())

    df_train.to_csv(temp_directory / "temp_train.csv", index=False)
    df_test.to_csv(temp_directory / "temp_test.csv", index=False)

    df_train = Datasetold(
        data_path=temp_directory / "temp_train.csv",
        features=feature_l,
        target=general_configuration["label"],
        configuration=general_configuration["processing"],
        is_train=True,
        experiment_path=MODELS_DIRECTORY / "temp",
    ).process_data()
    df_test = Datasetold(
        data_path=temp_directory / "temp_test.csv",
        features=feature_l,
        target=general_configuration["label"],
        configuration=general_configuration["processing"],
        is_train=False,
        experiment_path=MODELS_DIRECTORY / "temp",
    ).process_data()
    # clean the temp files
    shutil.rmtree(temp_directory)
    # Train the logistic regression on the remaining data
    x = df_train.data[feature_l].values
    x_test = df_test.data[feature_l].values
    y = df_train.data[general_configuration["label"]].values
    reg = LinearRegression().fit(x, y)
    log.info("LR coefficients:")
    log.info(reg.coef_)
    log.info("LR score:")
    log.info(reg.score(x, y))

    # Evaluate it on the test dataset
    pred = reg.predict(x_test)
    df_test.data["prediction"] = pred
    # TODO change the Evaluation class
    # Evaluation(
    #     df_test.data,
    #     "prediction",
    #     general_configuration["label"],
    #     MODELS_DIRECTORY / folder_name,
    #     general_configuration["comparison_score"],
    # )


def _check_local_config(local_config, single_model_name, general_configuration, benchmark_column):
    """Check config."""
    if next(iter(local_config["experiments"])) != single_model_name:
        raise Exception(f"Modular only makes sense with {single_model_name} - stopping")
    if local_config["label"] != general_configuration["label"]:
        raise Exception("Label not the same as in modular config - stopping")
    if local_config["benchmark_column"] != benchmark_column:
        raise Exception(
            "local config does not have same benchmark_column value as "
            + "main modular config - stopping"
        )
    if len(local_config["feature_paths"]) > 1:
        raise Exception("modular only works with one feature list per module - stopping")
    if len(local_config["models"]) > 1:
        raise Exception("modular only works with one model type per module - stopping")
