"""Data augmenter used to increase number of samples."""
import random

import click
import pandas as pd
from tqdm import tqdm

from ig import CONFIGURATION_DIRECTORY
from ig.utils.io import load_yml, read_data


def find_diff_pos(seq1: str, seq2: str) -> int:
    """Find position of the first difference between two strings."""
    indexes = range(len(seq1))
    for i in indexes:
        if seq1[i] != seq2[i]:
            break
    return i


genetic_code = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
}


@click.command()
@click.option("--configuration-file", "-c", type=str, required=True, help="name of the config file")
@click.option(
    "--augmentation-coeff",
    "-coeff",
    type=int,
    required=True,
    help="number of version to create per sequence",
)
@click.option("--dataset-path", "-data", type=str, required=True, help="Path to dataset file")
def augment(
    configuration_file: str,
    dataset_path: str,
    augmentation_coeff: int,
) -> None:
    """Augments given dataframe. Augmented version will have a balanced positive / negative ratio."""
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)

    dna_wt_col_name = general_configuration["general_params"]["dna_wildtype_col_name"]
    dna_mutated_col_name = general_configuration["general_params"]["dna_mutated_col_name"]

    peptide_wt_col_name = general_configuration["general_params"]["peptide_wildtype_col_name"]
    peptide_mutated_col_name = general_configuration["general_params"]["peptide_mutated_col_name"]

    df = read_data(dataset_path)
    pos_neg_ratio = len(df[df["cd8_any"] == 1]) / len(df[df["cd8_any"] == 0])
    wild_type_sequences = df[peptide_wt_col_name]
    mutated_sequences = df[peptide_mutated_col_name]
    labels = df["cd8_any"]
    mutation_start_positions = [
        find_diff_pos(wt, mutated) for wt, mutated in zip(wild_type_sequences, mutated_sequences)
    ]

    augmentation_rows = []
    for idx in range(len(df)):
        print(f"sample: {idx}/{len(df)}")
        tmp_row = df.iloc[[idx]].copy()
        wt_peptide = wild_type_sequences[idx]
        mutated_peptide = mutated_sequences[idx]

        mutation_start_position = mutation_start_positions[idx]
        if labels[idx] == 0:
            tmp_augmentation_coeff = int(augmentation_coeff * pos_neg_ratio)
        else:
            tmp_augmentation_coeff = augmentation_coeff
        for _ in tqdm(range(tmp_augmentation_coeff)):
            dna_wt = "".join([random.choice(genetic_code[base]) for base in wt_peptide])
            dna_mutated = "".join(
                [
                    dna_wt[: mutation_start_position * 3],
                    random.choice(genetic_code[mutated_peptide[mutation_start_position]]),
                    dna_wt[mutation_start_position * 3 + 3 :],
                ]
            )

            tmp_row[dna_wt_col_name] = dna_wt
            tmp_row[dna_mutated_col_name] = dna_mutated
            augmentation_rows.append(pd.DataFrame(tmp_row))

    augmentation_df = pd.concat(augmentation_rows + [df], ignore_index=True)
    augmentation_df = augmentation_df.sample(frac=1)
    file_name = dataset_path.split("/")[-1].split(".")[0]
    augmentation_df.to_csv(dataset_path.replace(file_name, f"augmented_{file_name}"))
