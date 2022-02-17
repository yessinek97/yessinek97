"""Test Units for TCR-pMHC data generation function."""
import os
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from tempfile import mktemp

import numpy as np
import pyrosetta as pr

from biondeep_ig.data_gen.ig.data_gen.rosetta import init_rosetta
from biondeep_ig.data_gen.ig.data_gen.rosetta import rechain
from biondeep_ig.data_gen.ig.data_gen.tcr_pmhc.tcr_pmhc_align import concatenate_poses
from biondeep_ig.data_gen.ig.data_gen.tcr_pmhc.tcr_pmhc_align import process_tm_mat
from biondeep_ig.data_gen.ig.data_gen.tcr_pmhc.tcr_pmhc_extract import extract_chain
from biondeep_ig.data_gen.ig.data_gen.tcr_pmhc.tcr_pmhc_extract import extract_tcr_pmhc_chains


def test_process_tm_mat(current_path: Path):
    """Check process_tm_mat numerically.

    Args:
        current_path: current path of the test
    """
    mat_path = current_path / "fixtures" / "tmalign" / "pmhc_peptide_align.txt"
    out_path = current_path / "fixtures" / "tmalign" / "pmhc_peptide_align_out.txt"
    t_expected = np.array([82.603858827, -41.126853376, -0.9479020751], dtype=np.float32)
    r_expected = np.array(
        [
            [0.3271227376, -0.800939933, -0.5014837368],
            [0.9448261723, 0.2675816802, 0.1889538266],
            [-0.0171528043, -0.5356260525, 0.8442810629],
        ],
        dtype=np.float32,
    )
    # lower th to get results
    t_got, r_got = process_tm_mat(
        mat_path=mat_path, out_path=out_path, use_th=True, tm_score_th=0.2
    )

    assert np.allclose(t_expected, t_got, rtol=1e-6)
    assert np.allclose(r_expected, r_got, rtol=1e-6)

    # use default th
    t_got, r_got = process_tm_mat(mat_path=mat_path, out_path=out_path, use_th=True)
    assert t_got is None
    assert r_got is None


def test_concatenate_poses(current_path: Path):
    """Check correct concatenation of PDBs into one pose.

    Arg:
        current_path: current path of the test
    """
    # Initialize Rosetta
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    temp_file_path = Path(mktemp())
    pmhc_path = current_path / "fixtures" / "protein_files" / "pmhc.pdb"
    tcr_path = current_path / "fixtures" / "protein_files" / "tcr.pdb"

    pmhc_pose = pr.pose_from_pdb(str(pmhc_path))
    tcr_pose = pr.pose_from_pdb(str(tcr_path))

    tcr_pose = rechain(pose=tcr_pose, chain_mapping={"A": "D", "B": "E"})

    assert {
        pr.rosetta.core.pose.get_chain_from_chain_id(chain_id=i, pose=pmhc_pose)
        for i in range(1, pmhc_pose.num_chains() + 1)
    } == {"A", "C"}

    assert {
        pr.rosetta.core.pose.get_chain_from_chain_id(chain_id=i, pose=tcr_pose)
        for i in range(1, tcr_pose.num_chains() + 1)
    } == {"D", "E"}

    concatenate_poses(
        poses=[pmhc_pose, tcr_pose],
        out_path=str(temp_file_path),
    )

    concat_pose = pr.pose_from_pdb(str(temp_file_path))

    assert {
        pr.rosetta.core.pose.get_chain_from_chain_id(chain_id=i, pose=concat_pose)
        for i in range(1, concat_pose.num_chains() + 1)
    } == {"A", "C", "D", "E"}

    os.remove(temp_file_path)


def test_rechain(current_path: Path):
    """Check correct renaming of chains.

    Skipped if PyRosetta is not installed.

    Args:
        current_path: current path of the test
    """
    # Initialize Rosetta
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    pdb_path = current_path / "fixtures" / "protein_files" / "insulin.pdb"
    pose = pr.pose_from_pdb(str(pdb_path))

    assert {
        pr.rosetta.core.pose.get_chain_from_chain_id(chain_id=i, pose=pose)
        for i in range(1, pose.num_chains() + 1)
    } == {"A", "B"}

    pose = rechain(pose=pose, chain_mapping={"B": "C"})

    assert {
        pr.rosetta.core.pose.get_chain_from_chain_id(chain_id=i, pose=pose)
        for i in range(1, pose.num_chains() + 1)
    } == {"A", "C"}


def test_extract_chain(current_path: Path):
    """Check correct chain extraction from PDB file.

    Args:
        current_path: current path of the test
    """
    # Initialize Rosetta
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    temp_dir_path = Path(mkdtemp())
    extracted_chain_path = temp_dir_path / "insulin_B.pdb"

    pdb_path = current_path / "fixtures" / "protein_files" / "insulin.pdb"
    pose = pr.pose_from_pdb(str(pdb_path))

    extract_chain(
        pose=pose,
        chain="B",
        file_path=extracted_chain_path,
    )

    extracted_pose = pr.pose_from_pdb(str(extracted_chain_path))

    assert extracted_pose.num_chains() == 1
    assert pr.rosetta.core.pose.get_chain_from_chain_id(chain_id=1, pose=extracted_pose) == "B"

    rmtree(temp_dir_path)


def test_extract_tcr_pmhc_chains(current_path: Path):
    """Check correct chains extraction from TCR-pMHC structures.

    Args:
        current_path: current path of the test
    """
    # Initialize Rosetta
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    temp_dir_path = Path(mkdtemp())
    pmhc_path = current_path / "fixtures" / "protein_files" / "pmhc.pdb"
    tcr_path = current_path / "fixtures" / "protein_files" / "tcr.pdb"
    tcr_pmhc_path = current_path / "fixtures" / "protein_files" / "tcr_pmhc.pdb"

    extract_tcr_pmhc_chains(
        pmhc_file=pmhc_path,
        tcr_file=tcr_path,
        template_file=tcr_pmhc_path,
        output_dir=temp_dir_path,
    )

    expected_files = [
        "pmhc_peptide.pdb",
        "tcr_beta.pdb",
        "tcr_rechained.pdb",
        "template_pmhc_peptide.pdb",
        "template_tcr_beta.pdb",
    ]

    assert set(expected_files) == set(os.listdir(str(temp_dir_path)))

    rmtree(temp_dir_path)
