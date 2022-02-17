"""Script for aggregating scores from pMHC pipeline."""
import argparse
import multiprocessing
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import pandas as pd
from tqdm import tqdm


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Parses rosetta scores.")
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        help="Path to pMHC pipeline output directory.",
        required=True,
    )
    parser.add_argument(
        "--out-path",
        "-o",
        type=str,
        help="Path to output aggregated scores file.",
        required=True,
    )
    args = parser.parse_args()

    return args


def parse_score(score_file: str) -> Union[List[Dict[str, Any]], None]:  # noqa: CCR001
    """Parse score from rosetta sc file.

    Args:
        score_file: path to score file.

    Returns:
        list fo field-value elements.
    """
    with open(score_file) as f:
        lines = [line.rstrip() for line in f.readlines()]

    try:
        labels = [score for score in lines[1].split("SCORE:")[1].split(" ") if score != ""]

        all_scores = []
        for i in range(2, len(lines)):
            if lines[i].startswith("SCORE:"):
                scores = []
                line = lines[i].split("SCORE:")[1].split(" ")
                for s in line:
                    if s != "":
                        scores.append(s)
                all_scores.append(scores)

        return [dict(zip(labels, scores)) for scores in all_scores]
    except IndexError:
        return None


def main():
    """Run main function."""
    args = parse_args()

    score_files = list(Path(args.dir).rglob("*.sc"))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        scores = list(tqdm(pool.map(parse_score, score_files)))

    scores = [score for score in scores if score is not None]

    all_scores = []
    for score in scores:
        all_scores.extend(score)

    pmhc_df = pd.DataFrame(all_scores)

    pmhc_df.to_csv(args.out_path, index=False)


if __name__ == "__main__":
    main()
