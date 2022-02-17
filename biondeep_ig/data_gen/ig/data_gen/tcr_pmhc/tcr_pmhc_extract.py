"""Extract chains from TCR, pMHC and TCR-pMHC template."""
import argparse
from pathlib import Path

import pyrosetta as pr

from biondeep_ig.data_gen.ig.data_gen.rosetta import init_rosetta
from biondeep_ig.data_gen.ig.data_gen.rosetta import rechain
from biondeep_ig.data_gen.ig.util import log

logger = log.get_logger(__name__)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Extract chains from TCR, pMHC and TCR-pMHC template."
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="PDB File containing TCR-pMHC template.",
        required=True,
    )
    parser.add_argument("--tcr", type=Path, help="TCR PDB file path.", required=True)
    parser.add_argument("--pmhc", type=Path, help="pMHC PDB file path.", required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Folder for output files.",
        required=True,
    )
    args = parser.parse_args()

    return args


def extract_chain(pose: pr.Pose, chain: str, file_path: Path):
    """Get one chain from the given pose.

    Args:
        pose: pyrosetta pose.
        chain: chain name.
        file_path: output path.
    """
    chain_indexes = set()
    for i in range(1, pose.total_residue()):
        c = pose.pdb_info().chain(i)
        if c == chain:
            chain_indexes.add(i)
    chain_pose = pr.Pose(pose, min(chain_indexes), max(chain_indexes))
    chain_pose.dump_pdb(str(file_path))


def extract_tcr_pmhc_chains(pmhc_file: Path, tcr_file: Path, template_file: Path, output_dir: Path):
    """Extract the following chains.

        - Peptide from pmhc_file (pMHC PDB)
        - TCR beta from tcr_file (TCR PDB)
        - Peptide from template_file (template pMHC-TCR)
        - TCR beta from template_file (template pMHC-TCR)

    We assume the following chain order in TCR-pMHC template:
    - A = MHCI-alpha
    - B = MHCI-beta
    - C = peptide
    - D = TCR-alpha
    - E = TCR-beta

    Args:
        pmhc_file: file path for the pMHC.
        tcr_file: file path for the TCR.
        template_file: file path for the TCR-pMHC template.
        output_dir: output dir
    """
    # Load complexed poses from PDBs
    pmhc = pr.pose_from_pdb(str(pmhc_file))  # pMHC PDB
    tcr = pr.pose_from_pdb(str(tcr_file))  # TCR PDB
    tpl = pr.pose_from_pdb(str(template_file))  # pMHC-TCR template PDB

    # Prepare tmp paths
    pep_path = output_dir / "pmhc_peptide.pdb"
    tcr_beta_path = output_dir / "tcr_beta.pdb"
    tcr_rechained_path = output_dir / "tcr_rechained.pdb"
    tpl_pep_path = output_dir / "template_pmhc_peptide.pdb"
    tpl_tcr_beta_path = output_dir / "template_tcr_beta.pdb"

    # Extract Peptide from pMHC complex
    if pmhc.pdb_info().num_chains() == 2:
        pmhc = rechain(pmhc, {"B": "C"})
        pmhc.dump_pdb(str(pmhc_file))
    extract_chain(pose=pmhc, chain="C", file_path=pep_path)

    # Extract TCR beta chain from TCR complex
    tcr = rechain(tcr, {"A": "D", "B": "E"})
    extract_chain(pose=tcr, chain="E", file_path=tcr_beta_path)
    tcr.dump_pdb(str(tcr_rechained_path))

    # Extract Peptide AND TCR beta chain from TCR-pMHC complex template
    extract_chain(pose=tpl, chain="C", file_path=tpl_pep_path)
    extract_chain(pose=tpl, chain="E", file_path=tpl_tcr_beta_path)


def main():
    """Main function."""
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    init_rosetta()
    extract_tcr_pmhc_chains(
        pmhc_file=args.pmhc,
        tcr_file=args.tcr,
        template_file=args.template,
        output_dir=args.output_dir,
    )
