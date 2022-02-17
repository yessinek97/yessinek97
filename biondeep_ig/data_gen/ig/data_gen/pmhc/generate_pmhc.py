"""Script to generate poses (Argo-based)."""
import argparse
import os
from pathlib import Path

import pyrosetta as pr

from biondeep_ig.data_gen.ig.data_gen.rosetta import build_min_mover
from biondeep_ig.data_gen.ig.data_gen.rosetta import build_score_function
from biondeep_ig.data_gen.ig.data_gen.rosetta import init_rosetta
from biondeep_ig.data_gen.ig.data_gen.rosetta import redock
from biondeep_ig.data_gen.ig.data_gen.rosetta import relax
from biondeep_ig.data_gen.ig.data_gen.rosetta import substitute_chain
from biondeep_ig.data_gen.ig.util import log

logger = log.get_logger(__name__)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Generate pMHC poses.")
    parser.add_argument(
        "--peptide",
        type=str,
        help="File containing peptide AA sequences.",
        required=True,
    )
    parser.add_argument("--init_pdb", type=Path, help="Init PDB path.", required=True)
    parser.add_argument("--flag", type=int, help="Pose Flag per pMHC.", required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="out/rosetta/",
        help="Folder for output files.",
    )
    parser.add_argument("--temperature", type=int, default=2, help="Temperature for Monte Carlo.")
    parser.add_argument(
        "--insert_pos_offset",
        type=int,
        default=3,
        help="offset position from where residues are added.",
    )
    parser.add_argument(
        "--pack_radius",
        type=int,
        default=8,
        help="distance threshold for peptide neighbours selection.",
    )
    args = parser.parse_args()

    return args


def generate_pmhc(
    pdb_path: str,
    peptide: str,
    output_path: str,
    temperature: int,
    num_extra_residues: int,
    insert_pos_offset: int = 3,
    pack_radius: int = 8,
):
    """Score the peptide.

    Args:
        pdb_path: path of the pdb file to load pyrosetta pose.
        peptide: AA sequence.
        output_path: output dir.
        temperature: temperature hyper-parameter.
        num_extra_residues: number of extra residues to add.
        insert_pos_offset: offset position from where residues are added.
        pack_radius: linked to the definition of neighbor.
    """
    # if file exists, do not execute
    if not os.path.exists(output_path + ".pdb"):
        score_fn = build_score_function()
        pose = pr.pose_from_pdb(pdb_path)
        pose = substitute_chain(
            score_fn=score_fn,
            pose=pose,
            seq=peptide,
            chain="C",
            start=1,
            temperature=temperature,
            num_extra_residues=num_extra_residues,
            pack_radius=pack_radius,
            insert_pos_offset=insert_pos_offset,
        )
        minmover = build_min_mover(score_fn=score_fn)
        pose = redock(score_fn=score_fn, minmover=minmover, pose=pose, temperature=temperature)
        min_pdb_path = f"{output_path}_min.pdb"
        pose.dump_pdb(min_pdb_path)
        pose = relax(score_fn=score_fn, pose=pose)
        relax_pdb_path = f"{output_path}_relax.pdb"
        pose.dump_pdb(relax_pdb_path)


def get_peptide_in_pose(file_path) -> str:
    """Get Peptide in Allele Init Structure.

    Args:
        file_path: path to template PDB.
    """
    pose = pr.pose_from_pdb(str(file_path))
    peptide_in_pose = pose.chain_sequence(2)
    return peptide_in_pose


def get_peptide_length(file_path) -> int:
    """Get Peptide Length.

    Args:
        file_path: path to template PDB.
    """
    return len(get_peptide_in_pose(file_path))


def check_peptide_length(file_path: str, peptide: str) -> int:
    """Compare lengths of target peptide and template's.

    Args:
        file_path: path to template PDB.
        peptide: target peptide.

    Returns:
        number of extra residues to be added to template, if any.

    Raises:
        ValueError if template peptide strictly shorter than peptide.
    """
    len_peptide = get_peptide_length(file_path=file_path)
    num_extra_residues = len(peptide) - len_peptide

    if num_extra_residues < 0:
        raise ValueError(
            (
                f"Mutant sequence is shorter ({len(peptide)} residues)"
                f" than the PDB sequence ({len_peptide} residues)"
            )
        )

    if num_extra_residues > 0:
        logger.warning(
            "Peptide %s has length %i, found length %i in PDB. "
            "%i residues will be added to the structure.",
            peptide,
            len(peptide),
            len_peptide,
            num_extra_residues,
        )

    return num_extra_residues


def main():
    """Main function."""
    args = parse_args()
    init_pdb_path = args.init_pdb
    # Get the name of the init pdb without the file extension.
    init_pdb_name = str(init_pdb_path.stem)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    init_rosetta()

    peptide = args.peptide
    num_extra_residues = check_peptide_length(file_path=init_pdb_path, peptide=peptide)

    output_path = str(output_dir / f"{init_pdb_name}_{peptide}_{args.flag}")
    generate_pmhc(
        str(init_pdb_path),
        peptide,
        output_path,
        args.temperature,
        num_extra_residues,
        args.insert_pos_offset,
        args.pack_radius,
    )


if __name__ == "__main__":
    main()
