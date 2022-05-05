"""Test the ablation study functions script."""
from pathlib import Path

import pytest

from biondeep_ig.ablation_study import get_command_line_for_combination
from biondeep_ig.ablation_study import get_features_from_combination
from biondeep_ig.ablation_study import write_configuration
from biondeep_ig.ablation_study import write_features
from biondeep_ig.src.utils import load_yml


@pytest.mark.parametrize(
    ("combination", "expected"),
    [
        (
            (0, 1, 1, 1),
            [
                "expression",
                "nontested_score_biondeep_mhci",
                "presentation_score",
                "tested_score_biondeep_mhci",
            ],
        ),
        (
            (1, 1, 1, 0),
            [
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
                "tested_kiderafac1_pep_mhci",
                "tested_pk_pep_mhci",
                "kytedoolittle_nontested_pep_moment_whole",
                "tested_score_biondeep_mhci",
                "gravy_nontested_pep_moment_mhci",
                "mcla720101",
                "tm_tend_tested_pep_global_whole",
                "nontested_foreignness_richman_pep_mhci",
            ],
        ),
    ],
)
def test_features_from_combination(combination, expected):
    """Test the content of the feature list for a given feature combination."""
    features = get_features_from_combination(combination)
    assert features == expected


def test_write_features():
    """Test that a given feature list is properly written."""
    features = [
        "expression",
        "nontested_score_biondeep_mhci",
        "presentation_score",
        "tested_score_biondeep_mhci",
    ]
    filename = Path(__file__).parent / "train_features.txt"

    write_features(features, filename)

    assert filename.is_file()

    with open(filename, "r") as f:
        content = [el.strip() for el in f.readlines()]

    assert content == features

    # Clean-up
    filename.unlink()


def test_write_configuration():
    """Test that the correct features list file is given in the training configuration file."""
    write_configuration("test_with_config.yml", "new_features.txt", init_config_file="train.yml")

    config_file = Path(__file__).parent.parent.parent / "configuration" / "test_with_config.yml"
    new_config = load_yml(config_file)
    assert new_config["feature_paths"] == ["new_features"]

    # Clean-up
    config_file.unlink()


def test_command_line():
    """Test the generation of the training command line for a given combination."""
    train_data = "train.csv"
    test_data = "test.csv"
    result = get_command_line_for_combination(
        train_data=train_data,
        test_data=test_data,
        combination=(1, 1, 1, 1),
        folder_name="ablation",
        init_config_file="train.yml",
    )
    config_file = "train_bnt_1_binding_1_expression_1_presentation_1.yml"

    expected = (
        f"train -train {train_data} -test {test_data} "
        f"-c {config_file} "
        "-n ablation/bnt_1_binding_1_expression_1_presentation_1"
    )

    assert result == expected

    (Path(__file__).parent.parent.parent / "configuration" / config_file).unlink()
    (
        Path(__file__).parent.parent.parent
        / "configuration/features/CD8/features_bnt_1_binding_1_expression_1_presentation_1.txt"
    ).unlink()
