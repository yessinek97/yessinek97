"""Align TCR and pMHC, then combine them."""
import argparse
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pyrosetta as pr

from biondeep_ig.data_gen.ig.data_gen.rosetta import init_rosetta
from biondeep_ig.data_gen.ig.util import log

logger = log.get_logger(__name__)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Align TCR and pMHC, then combine them.")
    parser.add_argument(
        "--pmhc_pdb",
        type=Path,
        help="pMHC PDB file.",
        required=True,
    )
    parser.add_argument(
        "--pmhc_tmalign",
        type=Path,
        help="pMHC rotation file.",
        required=True,
    )
    parser.add_argument(
        "--pmhc_tmalign_out",
        type=Path,
        help="pMHC TMalign console output.",
        required=True,
    )
    parser.add_argument(
        "--tcr_pdb",
        type=Path,
        help="TCR PDB file.",
        required=True,
    )
    parser.add_argument(
        "--tcr_tmalign",
        type=Path,
        help="TCR rotation file.",
        required=True,
    )
    parser.add_argument(
        "--tcr_tmalign_out",
        type=Path,
        help="TCR TMalign console output.",
        required=True,
    )
    parser.add_argument(
        "--out_pdb",
        type=Path,
        help="Output pMHC PDB file.",
        required=True,
    )
    args = parser.parse_args()

    return args


def process_tm_mat(
    mat_path: Path, out_path: Path, use_th: bool = False, tm_score_th: float = 0.5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Processes matrix.

    The file content is like following:
    ------ The rotation matrix to rotate Chain_1 to Chain_2 ------
    m               t[m]        u[m][0]        u[m][1]        u[m][2]
    0      82.6038588270   0.3271227376  -0.8009399330  -0.5014837368
    1     -41.1268533760   0.9448261723   0.2675816802   0.1889538266
    2      -0.9479020751  -0.0171528043  -0.5356260525   0.8442810629

    We need to extract this matrix

    The TMalign cmd print in console is like following (inlcuding an empty line at the beginning):

     *********************************************************************
     * TM-align (Version 20190822): protein structure alignment          *
     * References: Y Zhang, J Skolnick. Nucl Acids Res 33, 2302-9 (2005) *
     * Please email comments and suggestions to yangzhanglab@umich.edu   *
     *********************************************************************

    Name of Chain_1: tcr_beta.pdb (to be superimposed onto Chain_2)
    Name of Chain_2: template_tcr_beta.pdb
    Length of Chain_1: 109 residues
    Length of Chain_2: 245 residues

    Aligned length= 108, RMSD=   1.75, Seq_ID=n_identical/n_aligned= 0.315
    TM-score= 0.88486 (if normalized by length of Chain_1, i.e., LN=109, d0=3.84)
    TM-score= 0.41340 (if normalized by length of Chain_2, i.e., LN=245, d0=5.80)
    (You should use TM-score normalized by length of the reference structure)

    We need to extract the TM-score

    Args:
        mat_path: matrix file name
        out_path: console output file name
        use_th: whether to use a threshold criterion on TM score
        tm_score_th: threshold for maximum TM score, if the max < th, we should not use the results

    Returns:
        translation and rotation matrix
    """
    # parse the TMalign console print
    with open(out_path) as f:
        lines = f.read().splitlines()
    lines = [x for x in lines if x.startswith("TM-score=")]
    tm_scores = np.asarray([x.split()[1] for x in lines], dtype=np.float32)
    if use_th and np.max(tm_scores) < tm_score_th:
        # the score is too small
        logger.error("Could not align chains with threshold: %s", tm_score_th)
        return None, None

    # parse the output saved by -m
    with open(mat_path) as f:
        lines = f.read().splitlines()
    lines = lines[2:5]
    arr = np.asarray([x.split()[1:] for x in lines], dtype=np.float32)  # (3, 4)
    t = arr[:, 0]  # (3,)
    r = arr[:, 1:]  # (3, 3)
    return t, r


def concatenate_poses(poses: List[pr.Pose], out_path: str) -> None:
    """Merge poses into one single PDB file.

    Args:
        poses: list of pyrosetta poses
        out_path: path the single merged PDB
    """
    pdb_lines, pdb_conect_lines, ssbond_lines = [], [], []
    for i, pose in enumerate(poses):
        tmp_path = f"{str(Path(out_path).parent / Path(out_path).stem)}_{i}.pdb"
        pose.dump_pdb(tmp_path)
        with open(tmp_path) as f:
            all_lines = f.readlines()
            for line in all_lines:
                if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER"):
                    pdb_lines.append(line)
                elif line.startswith("SSBOND"):
                    ssbond_lines.append(line)
                elif line.startswith("CONECT"):
                    pdb_conect_lines.append(line)
        Path(tmp_path).unlink()
    with open(out_path, "w") as f:
        f.writelines(ssbond_lines)
        f.writelines(pdb_lines)
        f.writelines(pdb_conect_lines)


def main():
    """Main function."""
    args = parse_args()

    # Initialize Rosetta
    options = ["-rebuild_disulf false", "-detect_disulf false"]
    init_rosetta(options=options)

    # Load pMHC and TCR poses from respective PDBs
    pmhc = pr.pose_from_pdb(str(args.pmhc_pdb))
    tcr = pr.pose_from_pdb(str(args.tcr_pdb))

    # Load translation and rotations transforms from TMAlign outputs
    pmhc_mat_translate, pmhc_mat_rot = process_tm_mat(
        mat_path=args.pmhc_tmalign,
        out_path=args.pmhc_tmalign_out,
    )
    tcr_mat_translate, tcr_mat_rot = process_tm_mat(
        mat_path=args.tcr_tmalign,
        out_path=args.tcr_tmalign_out,
    )

    # If one of the alignment failed, stop and return None
    if pmhc_mat_translate is None:
        return
    if tcr_mat_translate is None:
        return

    # Apply rotations and translation to TCR and pMHC complexes
    pmhc.rotate(pmhc_mat_rot)
    pmhc.translate(pmhc_mat_translate)
    tcr.rotate(tcr_mat_rot)
    tcr.translate(tcr_mat_translate)

    # Reunite transformed poses into one Pose object
    # and save it
    concatenate_poses([pmhc, tcr], str(args.out_pdb))
