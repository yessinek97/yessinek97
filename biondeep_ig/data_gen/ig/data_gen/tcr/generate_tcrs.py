"""Script for generating TCRs from list of CDR3a-CDR3b sequences."""
import argparse
import subprocess
from pathlib import Path

from biondeep_ig.data_gen.ig.util import log

logger = log.get_logger(__name__)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Generate TCR structures.")
    parser.add_argument(
        "--tcrmodel-path",
        "-e",
        type=str,
        default="/home/app/biondeep_ig/data_gen/ig/data_gen/tcr/tcrmodel.sh",
        help="Path to tcrmodel.sh executable.",
    )
    parser.add_argument(
        "--tcrs-path",
        "-t",
        type=str,
        help="Path to text file containing list of CDR3a-CDR3b sequences.",
        required=True,
    )
    parser.add_argument(
        "--out-dir-path",
        "-o",
        type=Path,
        help="Path to output directory.",
        required=True,
    )
    args = parser.parse_args()

    return args


def generate_tcr(tcrmodel_path: str, cdr3a: str, cdr3b: str, out_dir_path: Path):
    """Run Rosetta tcrmodel subprocess.

    Args:
        tcrmodel_path: path to tcrmodel.sh executable.
        cdr3a: CDR3a sequence of the target TCR.
        cdr3b: CDR3b sequence of the target TCR.
        out_dir_path: path to ouput directory.
    """
    try:
        subprocess.run(
            [
                tcrmodel_path,
                f"-a {cdr3a}",
                f"-b {cdr3b}",
                f"-o {str(out_dir_path)}",
            ],
            check=True,
        )
        logger.info(
            "TCR %s-%s has been generated.",
            cdr3a,
            cdr3b,
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "TCR modelling failed for CDR3a: %s and CDR3: %s.",
            cdr3a,
            cdr3b,
        )


def main():
    """Generates TCRs from list."""
    args = parse_args()
    output_dir_path: Path = args.out_dir_path

    with open(args.tcrs_path) as file:
        tcrs = [tcr.rstrip() for tcr in file.readlines()]

    for tcr in tcrs:
        alpha, beta = tcr.split("-")
        out_path = output_dir_path / f"tcr-{alpha}-{beta}"
        out_path.mkdir(exist_ok=True, parents=True)
        generate_tcr(args.tcrmodel_path, alpha, beta, out_path)


if __name__ == "__main__":
    main()
