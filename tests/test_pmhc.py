"""Test Units for pMHC data generation function."""
import re
from pathlib import Path

import pytest

from biondeep_ig.data_gen.ig.data_gen.pmhc.generate_pmhc import check_peptide_length
from biondeep_ig.data_gen.ig.data_gen.pmhc.generate_pmhc import get_peptide_in_pose
from biondeep_ig.data_gen.ig.data_gen.pmhc.generate_pmhc import get_peptide_length
from biondeep_ig.data_gen.ig.data_gen.rosetta import init_rosetta


def test_get_peptide_in_pose(current_path: Path):
    """Check correct extraction of peptide in pose.

    Arg:
        current_path: current path of the test
    """
    # Initialize Rosetta
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    pmhc_path = current_path / "fixtures" / "protein_files" / "pmhc.pdb"

    peptide = get_peptide_in_pose(file_path=pmhc_path)

    assert peptide == "AIMPARFYP"


def test_get_peptide_length(current_path: Path):
    """Check correct length of extracted peptide from pose.

    Arg:
        current_path: current path of the test
    """
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    pmhc_path = current_path / "fixtures" / "protein_files" / "pmhc.pdb"

    peptide_len = get_peptide_length(file_path=pmhc_path)

    assert peptide_len == 9


def test_check_peptide_length(current_path: Path):
    """Check correct length of extracted peptide from pose.

    Arg:
        current_path: current path of the test
    """
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    pmhc_path = str(current_path / "fixtures" / "protein_files" / "pmhc.pdb")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mutant sequence is shorter (8 residues) than the PDB sequence (9 residues)"
        ),
    ):
        _ = check_peptide_length(file_path=pmhc_path, peptide="A" * 8)

    assert check_peptide_length(file_path=pmhc_path, peptide="A" * 12) == 3
    assert check_peptide_length(file_path=pmhc_path, peptide="A" * 9) == 0
