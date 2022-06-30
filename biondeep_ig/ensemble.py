"""Command lines to aggregate experiments and checkpoints using mean and max."""
from pathlib import Path

import click
import numpy as np
import pandas as pd

from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig import ROOT_DIRECTORY
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.metrics import topk
from biondeep_ig.src.utils import load_yml

log = get_logger("Ensemble methods")


@click.command()
@click.option(
    "--exp_folder_path",
    "-e",
    type=str,
    required=False,
    multiple=True,
    help="Path to the folder expriments used in training.",
)
def ensemblexprs(exp_folder_path):
    """Generate results by ensembling existing experiments.

    This method supports ensembling of # experiments.
    looping over all best predicitions for each experiment
    and aggregate the checkpoints by the mean.

    Args:
        exp_folder_path: path to a folder
        which contains all the experiements.
        exp_files_path: path to the experiments.
        Note: You can use one argument at a time.

    Retruns:
        Mean topk of the ensembling method.
    """
    init_logger(ROOT_DIRECTORY, "Ensembling of multiple Experiment")
    if not exp_folder_path:
        exp_folder_path = Path("models").iterdir()
        log.info("looping over the models folder")
    else:
        exp_folder_path = [MODELS_DIRECTORY / path for path in exp_folder_path]

    mean_preds = []
    for experiment in exp_folder_path:
        file = pd.read_csv(Path(experiment) / "best_experiment/prediction/test.csv")
        file = file.dropna(subset=["cd8_any"])
        configuration = load_yml(Path(experiment) / "best_experiment" / "configuration.yml")
        preds_name = configuration["evaluation"]["prediction_name_selector"]
        mean_preds.append(file[preds_name])
    labels = file["cd8_any"]
    mean_preds_final = np.mean(mean_preds, axis=0)
    log.info(f"Top K mean : {topk(labels, mean_preds_final)}")


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
        one_exp: path to the experiement.

    Retruns:
        topk of the ensembling method.
    """
    init_logger(one_exp, "Ensembling of one Experiment")
    one_exp = MODELS_DIRECTORY / one_exp
    if (
        not (Path(one_exp) / "KfoldExperiment").exists()
        and not (Path(one_exp) / "SingleModel").exists()
    ):
        raise Exception(
            "No Experiments found! Please check that \
                        The experiment has Single Mode and/or K Fold Experiment Folder!"
        )

    topk_single, topk_kfold = 0, 0
    if (Path(one_exp) / "SingleModel").exists():
        topk_single = generate_topk_for_exp(Path(one_exp) / "SingleModel")
        log.info(f"Top K mean Single: {topk_single}")

    if (Path(one_exp) / "KfoldExperiment").exists():
        topk_kfold = generate_topk_for_exp(Path(one_exp) / "KfoldExperiment")
        log.info(f"Top K mean Kfold experiment: : {topk_kfold}")

    log.info(f"Top K mean, max Kfold and Single experiment: : {max(topk_kfold,topk_single)}")


def generate_topk_for_exp(folder_path, prediction_name=None) -> int:
    """Generate the top k for a single of kfold experiment.

    This method calculates the mean prediction for a single
    or kfold experiment across different features
    and differnet base models ie (xgboost or LGBM)

    Args:
        folder_path: path to experiement
        and the type of experiment used single of kfold.
        prediction_name: prediction in case of  using a single model
        and prediction_mean in case of using a kfold model.

    Retruns:
        Mean topk of the ensembling method.

    """
    mean_preds = []
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
