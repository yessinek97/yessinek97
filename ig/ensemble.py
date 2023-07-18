"""Command lines to aggregate experiments and checkpoints using mean and max."""
from logging import Logger
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd

from ig import MODELS_DIRECTORY
from ig.src.logger import get_logger, init_logger
from ig.src.metrics import topk
from ig.src.utils import load_yml

log: Logger = get_logger("Ensemble methods")


@click.command()
@click.option(
    "--exp_folder_path",
    "-e",
    type=str,
    required=False,
    multiple=True,
    help="Path to the folder expriments used in training.",
)
def ensemblexprs(exp_folder_name: str) -> None:
    """Generate results by ensembling existing experiments.

    This method supports ensembling of # experiments.
    looping over all best predicitions for each experiment
    and aggregate the checkpoints by the mean.

    Args:
        exp_folder_name: path to a folder
        which contains all the experiements.
        exp_files_path: path to the experiments.
        Note: You can use one argument at a time.

    Returns:
        Mean topk of the ensembling method.
    """
    init_logger(MODELS_DIRECTORY / "Ensambling", "EnsemblingMExp")
    if not exp_folder_name:
        exp_folder_path = list(Path("models").iterdir())
        log.info("looping over the models folder")
    else:
        exp_folder_path = [MODELS_DIRECTORY / path for path in exp_folder_name]

    mean_preds: List[np.array] = []
    for experiment in exp_folder_path:
        file = pd.read_csv(Path(experiment) / "best_experiment/prediction/test.csv")
        file = file.dropna(subset=["cd8_any"])
        configuration = load_yml(Path(experiment) / "best_experiment" / "configuration.yml")
        preds_name = configuration["evaluation"]["prediction_name_selector"]
        mean_preds.append(file[preds_name])
    labels = file["cd8_any"]
    mean_preds_final = np.mean(mean_preds, axis=0)
    log.info("Top K mean : %s", topk(labels, mean_preds_final))


@click.command()
@click.option(
    "--one_exp",
    "-s",
    type=str,
    required=True,
    help="Path to the folder expriments used in training",
)
def ensoneexp(one_exp: Path) -> None:
    """Generate results by ensembling an existing experiment.

    This method supports ensembling of one experiment.
    looping over all the features  and model (Xgboost/LGBM) results
    and aggregate the checkpoints by the mean.

    Args:
        one_exp: path to the experiment.

    Returns:
        topk of the ensembling method.
    """
    one_exp = MODELS_DIRECTORY / one_exp
    init_logger(one_exp, "Ensembling")

    if (
        not (Path(one_exp) / "KfoldExperiment").exists()
        and not (Path(one_exp) / "SingleModel").exists()
    ):
        raise Exception(
            "No Experiments found! Please check that \
                        The experiment has Single Mode and/or K Fold Experiment Folder!"
        )

    topk_single, topk_kfold = 0.0, 0.0
    if (Path(one_exp) / "SingleModel").exists():
        topk_single = generate_topk_for_exp(Path(one_exp) / "SingleModel")
        log.info("Top K mean Single %s:", topk_single)

    if (Path(one_exp) / "KfoldExperiment").exists():
        topk_kfold = generate_topk_for_exp(Path(one_exp) / "KfoldExperiment")
        log.info("Top K mean Kfold experiment %s:", topk_kfold)

    log.info("Top K mean, max Kfold and Single experiment %s:", max(topk_kfold, topk_single))


def generate_topk_for_exp(folder_path: Path, prediction_name: Optional[str] = None) -> float:
    """Generate the top k for a single of kfold experiment.

    This method calculates the mean prediction for a single
    or kfold experiment across different features
    and different base models ie (xgboost or LGBM)

    Args:
        folder_path: path to experiment
        and the type of experiment used single of kfold.
        prediction_name: prediction in case of  using a single model
        and prediction_mean in case of using a kfold model.

    Returns:
        Mean topk of the ensembling method.

    """
    mean_preds: List[np.array] = []
    for exp_feature in folder_path.iterdir():
        for model_type in (folder_path / exp_feature).iterdir():
            file = pd.read_csv(folder_path / exp_feature / model_type / "prediction/test.csv")
            file = file.dropna(subset=["cd8_any"])
            labels = file["cd8_any"]
            configuration = load_yml(folder_path / exp_feature / model_type / "configuration.yml")
            if not prediction_name:
                prediction_name = configuration["evaluation"]["prediction_name_selector"]
            mean_preds.append(file[prediction_name])
    labels = file["cd8_any"]
    mean_preds_final = np.mean(mean_preds, axis=0)

    return topk(labels, mean_preds_final)
