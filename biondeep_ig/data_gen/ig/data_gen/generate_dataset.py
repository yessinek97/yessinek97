"""Script to generate dataset for pMHC-TCR structures.

Example:
python3 generate_dataset.py \
    -d /data/public_ig.csv /data/optima_ig.csv \
    -p /data/pmhcs/ \
    -n 2
    --tm-alignment-path /data/tcrs_alignments.csv
"""
import argparse
from pathlib import Path

from biondeep_ig.data_gen.ig.data_gen.generate_pmhc_dataset import generate_pmhc_dataset
from biondeep_ig.data_gen.ig.data_gen.generate_tcr_pmhc_dataset import generate_tcr_pmhc_dataset


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Generate pMHC poses.")
    parser.add_argument(
        "--datasets",
        "-d",
        type=str,
        nargs="+",
        help="Paths to dataset csvs.",
        required=True,
    )
    parser.add_argument("--pmhc-dir", "-p", type=str, help="Path to pMHC dir.", required=True)
    parser.add_argument("--num-flags", "-n", type=int, help="Number of flags per pMHC.", default=4)
    parser.add_argument(
        "--artefacts-dir",
        type=str,
        help="Path to artefacts in pipeline.",
        default="/mnt/data/ig-dataset/artefacts",
    )
    parser.add_argument(
        "--out-pmhc-data-path", type=str, help="Path to out pMHC data csv.", default="pmhc_data.csv"
    )

    parser.add_argument(
        "--tm-alignment-path",
        type=str,
        help="Path to TCR-template alignment csv.",
        default="tm_alignment.csv",
    )
    parser.add_argument(
        "--out-tcr-pmhc-data-path",
        type=str,
        help="Path to out TCR-pMHC data csv.",
        default="tcr_pmhc_data.csv",
    )

    args = parser.parse_args()

    return args


def main():
    """Main."""
    args = parse_args()

    pmhc_df = generate_pmhc_dataset(
        paths=args.datasets,
        pmhc_dir=args.pmhc_dir,
        num_flags=args.num_flags,
        pipeline_prefix_dir=Path(args.artefacts_dir) / "pmhc",
    )
    pmhc_df.to_csv(args.out_pmhc_data_path, index=False)

    tcr_pmhc_df = generate_tcr_pmhc_dataset(
        pmhc_df=pmhc_df,
        tm_alignment_path=args.tm_alignment_path,
        artefacts_dir=args.artefacts_dir,
    )
    tcr_pmhc_df.to_csv(args.out_tcr_pmhc_data_path, index=False)


if __name__ == "__main__":
    main()
