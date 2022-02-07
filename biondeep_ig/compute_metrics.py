"""Module used to define the inference on a new test set command."""
import copy
from pathlib import Path

import click

import biondeep_ig.src.experiments as exper
from biondeep_ig.src import MODELS_DIRECTORY
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.utils import get_best_experiment
from biondeep_ig.src.utils import get_model_module_by_name
from biondeep_ig.src.utils import load_experiments
from biondeep_ig.src.utils import load_models
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import log_summary_results

log = get_logger("Eval")


@click.command()  # noqa
@click.option(
    "--test_data_paths",
    "-test",
    type=str,
    required=True,
    multiple=True,
    help="Path to the dataset used in  evaluation.",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
def compute_metrics(test_data_paths, folder_name):
    """Evaluation  on a seperate datasets."""
    init_logger(folder_name=folder_name, file_name="InfoEval")
    experiment_path = MODELS_DIRECTORY / folder_name
    general_configuration = load_yml(experiment_path / "configuration.yml")
    eval_configuration = general_configuration["evaluation"]

    log.info("****************************** Load YAML ****************************** ")
    experiment_names, experiment_params = load_experiments(general_configuration)
    log.info("****************************** Load EXP ****************************** ")
    model_types, model_params = load_models(general_configuration)
    log.info("****************************** Load Models ****************************** ")
    features_list_paths = copy.deepcopy(general_configuration["feature_paths"])
    log.info("****************************** Load Features lists *****************************")
    for test_data_path in test_data_paths:

        file_name = Path(test_data_path).stem
        log.info(f"{'#'* 20} \n")
        log.info(f"Start evaluating {file_name}\n")
        log.info(f"{'#'* 20} \n")

        results = []
        for experiment_name, experiment_param in zip(experiment_names, experiment_params):
            log.info(f"{experiment_name} :")
            experiment_class = get_model_module_by_name(exper, experiment_name)
            for model_type, _ in zip(model_types, model_params):
                log.info(f" {model_type} :")
                for features_list_path in features_list_paths:
                    features_list_name = features_list_path
                    run_path = experiment_path / experiment_name / features_list_name / model_type
                    configuration = load_yml(run_path / "configuration.yml")
                    log.info(f"  {features_list_name} :")
                    experiment = experiment_class(
                        train_data_path=None,
                        test_data_path=test_data_path,
                        configuration=configuration,
                        experiment_name=experiment_name,
                        folder_name=folder_name,
                        sub_folder_name=features_list_name,
                        **experiment_param,
                    )
                    scores = experiment.eval_test()
                    results.append(scores)
        _, eval_message = get_best_experiment(
            results, eval_configuration, path=MODELS_DIRECTORY / folder_name, file_name="results"
        )
        log_summary_results(eval_message)

        log.info("Eval best experiment")
        best_exp_path = experiment_path = MODELS_DIRECTORY / folder_name / "best_experiment"
        best_exp_configuration = load_yml(best_exp_path / "configuration.yml")
        best_experiment_name = list(best_exp_configuration["experiments"])[0]
        best_experiment_param = best_exp_configuration["experiments"][best_experiment_name]
        experiment = experiment_class(
            train_data_path=None,
            test_data_path=test_data_path,
            configuration=best_exp_configuration,
            experiment_name=best_experiment_name,
            folder_name=folder_name,
            sub_folder_name=best_exp_configuration["features"],
            experiment_directory=best_exp_path,
            **best_experiment_param,
        )
        log.info(f"Experiment : {best_experiment_name}")
        log.info(f"model : {best_exp_configuration['model_type']}")
        log.info(f"features : {best_exp_configuration['features']}")

        scores = experiment.eval_test()
