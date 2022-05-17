"""Test the merging of tcr-pMHC data to a given dataset."""
import random
import shutil
import string
from pathlib import Path

import numpy as np
import pandas as pd

from biondeep_ig.data_gen.ig.data_gen import merge_tcrpmhc_to_dataset

letters = string.ascii_lowercase
peptides = ["".join(random.choice(letters) for i in range(10)) for i in range(15)]
alleles = ["".join(random.choice(letters) for i in range(10)) for i in range(15)]

TCR_COLUMNS = [
    "sc_value",
    "complex_normalized",
    "dG_cross",
    "dG_cross/dSASAx100",
    "dG_separated",
    "dG_separated/dSASAx100",
    "dSASA_hphobic",
    "dSASA_polar",
    "dSASA_int",
    "delta_unsatHbonds",
    "hbond_E_fraction",
    "hbonds_int",
    "nres_all",
    "nres_int",
    "packstat",
    "per_residue_energy_int",
    "side1_score",
    "side2_score",
    "side1_normalized",
    "side2_normalized",
    "total_score",
    "description",
]

EXPECTED_MERGED_COLUMNS = [
    "sc_value",
    "complex_normalized",
    "dg_separated",
    "dg_separated/dsasax100",
    "dsasa_hphobic",
    "dsasa_polar",
    "dsasa_int",
    "delta_unsatHbonds",
    "hbond_e_fraction",
    "hbonds_int",
    "nres_all",
    "nres_int",
    "per_residue_energy_int",
    "side1_score",
    "side2_score",
    "side1_normalized",
    "side2_normalized",
]


def generate_dummy_data():
    """Generate random to represent training or testing datasets and save it to csv file.

    Returns:
        str: Path to the saved csv file.
    """
    df = pd.DataFrame(
        np.random.randint(-100, 100, size=(15, 2)) / 100, columns=["dummy1", "dummy2"]
    )

    df["tested_peptide_biondeep_mhci"] = peptides
    df["tested_allele_biondeep_mhci"] = alleles
    filename = "dummy_data.csv"
    df.to_csv(filename, index=False)
    return filename


def generate_dummy_tcr_pmhc_data():
    """Generate random to represent tcr-pMHC datasets and save it to csv file.

    Returns:
        str: Path to the saved csv file.
    """
    tcr_data = {}
    tcr_data["peptide"] = []
    tcr_data["allele"] = []
    for col in TCR_COLUMNS:
        tcr_data[col] = []

    for peptide, allele in zip(peptides, alleles):
        nb_structures = np.random.randint(3, 75)
        for _ in range(nb_structures):
            tcr_data["peptide"].append(peptide)
            tcr_data["allele"].append(allele)
            for col in TCR_COLUMNS:
                tcr_data[col].append(np.random.rand())

    tcr_data = pd.DataFrame(tcr_data)
    tcr_data["allele-peptide"] = tcr_data["allele"] + "-" + tcr_data["peptide"]
    # Negative values
    tcr_data["dG_separated"] = -1.0 * tcr_data["dG_separated"]
    tcr_data["dG_separated/dSASAx100"] = -1.0 * tcr_data["dG_separated/dSASAx100"]

    filename = "tcr_pmhc_dummy.csv"
    tcr_data.to_csv(filename, index=False)
    return filename


def test_merge_tcr_pmhc_data():
    """Test the function merging of tcr-pMHC data to a given dataset."""
    dummy_data_file = generate_dummy_data()
    dummy_tcr_data_file = generate_dummy_tcr_pmhc_data()

    merged_data_path = Path().cwd() / "merged_data"
    merged_data_path.mkdir(exist_ok=True)
    merge_tcrpmhc_to_dataset.main(
        input_dataset=dummy_data_file,
        tcr_pmhc_dataset=dummy_tcr_data_file,
        output_dir=str(merged_data_path),
    )

    for merged_file in merged_data_path.rglob("*.csv"):
        merged = pd.read_csv(merged_file)
        for col in EXPECTED_MERGED_COLUMNS:
            assert (
                col.lower() in merged.columns
            ), f"{col.lower()} is missing in {merged.columns.to_list()}"
        assert np.all(merged.isna().sum() == 0)

    # Clean-up
    Path(dummy_data_file).unlink()
    Path(dummy_tcr_data_file).unlink()
    shutil.rmtree(merged_data_path)
