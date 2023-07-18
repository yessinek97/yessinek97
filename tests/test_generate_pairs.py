# type: ignore
"""This module includes the tests for generate peptide allele  pairs module."""
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from ig.generate_pairs_peptide_allele import generate_pairs


@mock.patch("click.confirm")
def test_generate_pairs(
    mock_click: mock.MagicMock,
    train_data_path: str,
    generate_peptide_allele_output: str,
    local_path: str,
    peptide: str = "tested_peptide_biondeep_mhci",
    allele: str = "tested_allele_biondeep_mhci",
) -> None:
    """This function tests the behavior of generate peptide allele pairs."""
    mock_click.return_value = "y"

    runner = CliRunner()
    runner = CliRunner()
    params = [
        "-d",
        train_data_path,
        "-p",
        peptide,
        "-a",
        allele,
        "-o",
        generate_peptide_allele_output,
    ]
    _ = runner.invoke(generate_pairs, params)
    assert (
        Path(local_path) / "generated_pepetide_allele.csv"
    ).exists(), "Generate peptide allele file doesn't exist!"
