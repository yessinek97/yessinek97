"""Runs the ablation study: IG model training for different feature combination."""
import itertools
import subprocess
from copy import deepcopy
from pathlib import Path

import click
import pandas as pd

from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import save_yml

ALL_FEATURES = [
    "wt_fasgai_alpha_turn",
    "expression",
    "tested_dissimilarity_richman_pep_mhci",
    "nontested_score_biondeep_mhci",
    "gravy_nontested_pep_moment_whole",
    "nontested_kiderafac1_pep_mhci",
    "eisenberg_tested_pep_moment_whole",
    "eisenberg_tested_pep_moment_mhci",
    "tested_fasgai_alpha_turn_pep_mhci",
    "argos_nontested_pep_moment_whole",
    "nontested_charge_stryer_pep_mhci",
    "mswhim1_tc3_mhci",
    "nontested_kiderafac10_pep_mhci",
    "presentation_score",
    "tested_kiderafac1_pep_mhci",
    "tested_pk_pep_mhci",
    "kytedoolittle_nontested_pep_moment_whole",
    "tested_score_biondeep_mhci",
    "gravy_nontested_pep_moment_mhci",
    "mcla720101",
    "tm_tend_tested_pep_global_whole",
    "nontested_foreignness_richman_pep_mhci",
]
BNT_FEATURES = [
    "wt_fasgai_alpha_turn",
    "tested_dissimilarity_richman_pep_mhci",
    "gravy_nontested_pep_moment_whole",
    "nontested_kiderafac1_pep_mhci",
    "eisenberg_tested_pep_moment_whole",
    "eisenberg_tested_pep_moment_mhci",
    "tested_fasgai_alpha_turn_pep_mhci",
    "argos_nontested_pep_moment_whole",
    "nontested_charge_stryer_pep_mhci",
    "mswhim1_tc3_mhci",
    "nontested_kiderafac10_pep_mhci",
    "tested_kiderafac1_pep_mhci",
    "tested_pk_pep_mhci",
    "kytedoolittle_nontested_pep_moment_whole",
    "gravy_nontested_pep_moment_mhci",
    "mcla720101",
    "tm_tend_tested_pep_global_whole",
    "nontested_foreignness_richman_pep_mhci",
]
BINDING = ["nontested_score_biondeep_mhci", "tested_score_biondeep_mhci"]
EXPRESSION = ["expression"]
PRESENTATION = ["presentation_score"]

log = get_logger("Ablation study")


def get_features_from_combination(combination):
    """Construct a features list from a given combination of features.

    Args:
        combination (tuple): A 4-tuple indicating which features are taken.

    Returns:
        list(str): A list of features.
    """
    features = deepcopy(ALL_FEATURES)
    if combination[0] == 0:
        for feat in BNT_FEATURES:
            features.remove(feat)
    if combination[1] == 0:
        for feat in BINDING:
            features.remove(feat)
    if combination[2] == 0:
        for feat in EXPRESSION:
            features.remove(feat)
    if combination[3] == 0:
        for feat in PRESENTATION:
            features.remove(feat)
    return features


def write_features(features, filename):
    """Write the features list to the features file.

    Features filename must end with .txt

    Args:
        features (list(str)): List of features.
        filename (str or Path): Features file name.
    """
    if not str(filename).endswith(".txt"):
        raise ValueError(f"filename argument must end with '.txt' (got instead: {filename}")
    features_path = Path("configuration/features/CD8") / filename
    with open(str(features_path), "w") as f:
        for feat in features:
            f.write(feat + "\n")


def write_configuration(config_file, features_file, init_config_file):
    """Write a configuration file corresponding to a given features combination.

    Args:
        config_file (str): Path to the output configuration file.
        features_file (str): Path to the features list.
        init_config_file (str, optional): Initial file from which the configuration is build.

    Raises:
        ValueError: If the output config file name does not have .yml extension.
    """
    train_config_path = Path("configuration")
    config = load_yml(train_config_path / init_config_file)
    config["feature_paths"] = [str(Path(features_file).stem)]

    if not str(config_file).endswith(".yml"):
        raise ValueError(f"filename argument must end with '.yml' (got instead: {config_file}")
    save_yml(config, train_config_path / config_file)


def get_command_line_for_combination(
    train_data, test_data, combination, folder_name, init_config_file
):
    """Generate command line that executes the model training for a given features combination.

    Args:
        train_data (str): Path to train dataset.
        test_data (str): Path to test dataset.
        combination (tuple): Features combination.
        folder_name (str): Name of experiment.
        init_config_file (str, optional): Initial file from which the configuration is build.
    """
    combination_name = (
        f"bnt_{combination[0]}"
        f"_binding_{combination[1]}"
        f"_expression_{combination[2]}"
        f"_presentation_{combination[3]}"
    )

    features = get_features_from_combination(combination)
    features_file = f"features_{combination_name}.txt"
    write_features(features, features_file)

    config_file = f"train_{combination_name}.yml"
    write_configuration(config_file, features_file, init_config_file)

    cmd_line = (
        f"train -train {train_data} -test {test_data} "
        f"-c {config_file} "
        f"-n {folder_name}/{combination_name}"
    )
    return cmd_line


@click.command()
@click.option(
    "--train_data_path",
    "-train",
    type=str,
    required=True,
    help="Path to the training dataset.",
)
@click.option(
    "--test_data_path",
    "-test",
    type=str,
    required=True,
    help="Path to the testing dataset.",
)
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
@click.option(
    "--configuration_file",
    "-c",
    type=str,
    required=True,
    help="Path to the template configuration file.",
)
@click.option(
    "--nprocs",
    type=int,
    required=False,
    default=1,
    help="Number of processes run in parallel. Defaults to 1.",
)
def main(train_data_path, test_data_path, folder_name, configuration_file, nprocs=1):
    """Execute the ablation study based with the given datasets and based on a template configuration file.

    Can be run in parallel.

    Args:
        train_data_path (str): Path to the training dataset.
        test_data_path (str): Path to the testing dataset.
        folder_name (str): Experiment name.
        configuration_file (str): Path to the template configuration file.
        nprocs (int, optional): Number of processes run in parallel. Defaults to 1.
    """
    init_logger()

    # All combinations that include BNT features.
    ablation_combinations = list(itertools.product((1,), (0, 1), (0, 1), (0, 1)))

    # No BNT features but all the rest.
    ablation_combinations.insert(4, (0, 1, 1, 1))

    cmd_lines = []
    for combi in ablation_combinations:
        cmd_lines.append(
            get_command_line_for_combination(
                train_data_path, test_data_path, combi, folder_name, configuration_file
            )
        )

    print(nprocs)
    if nprocs == 1:
        for cmd in cmd_lines:
            log.info(f"Executing: {cmd}")
            subprocess.check_call(cmd.split())
    else:
        with open("commands.txt", "w") as f:
            for cmd in cmd_lines:
                f.write(cmd + "\n")
        parallel_cmd = f"parallel --jobs {nprocs} :::: commands.txt"
        log.info(f"Executing: {parallel_cmd}")
        subprocess.check_call(parallel_cmd.split())


@click.command()
@click.option("--folder_name", "-n", type=str, required=True, help="Experiment name.")
def postprocess(folder_name):
    """Postprocess an ablation study and print out the global topk and the topk per patient for all best experiments.

    Args:
        folder_name (str): Experiment name.
    """
    runs = []
    for path in Path(f"models/{folder_name}").iterdir():
        if path.is_dir():
            runs.append(path)

    metrics_selector = "topk"
    metrics_selector_higher = True
    dfs = []
    for run in runs:
        result_file = run / "results.csv"
        print(f"processing: {result_file}")
        df = pd.read_csv(result_file)
        df = df.loc[
            df["split"].str.contains("^test$"),
            [metrics_selector, "topk_20_patientid", "experiment", "model", "features"],
        ]
        df = df.sort_values(metrics_selector, ascending=metrics_selector_higher)
        df[metrics_selector] = df[metrics_selector].apply(lambda x: f"{x:.3f}")
        df["topk_20_patientid"] = df["topk_20_patientid"].apply(lambda x: f"{x:.3f}")
        df = df.iloc[-1:]
        dfs.append(df)

    ablation_results = pd.concat(dfs)
    ablation_results[["bnt", "binding", "expression", "presentation"]] = ablation_results[
        "features"
    ].str.extract(r"bnt_(\d)_binding_(\d)_expression_(\d)_presentation_(\d)")
    ablation_results = ablation_results.sort_values("features", ascending=False)
    print(ablation_results.drop("features", axis=1))
