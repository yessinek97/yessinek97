"""Data augmenter used to increase number of samples."""
import random

import click
import pandas as pd
from tqdm import tqdm

from ig import CONFIGURATION_DIRECTORY
from ig.utils.io import load_yml, read_data
from ig.utils.logger import get_logger

log = get_logger("data/augmentation")


def find_diff_pos(seq1: str, seq2: str) -> int:
    """Find position of the first difference between two strings."""
    indexes = range(len(seq1))
    for i in indexes:
        if seq1[i] != seq2[i]:
            break
    return i


genetic_code_dna = {
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

genetic_code_rna = {
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "C": ["UGU", "UGC"],
    "D": ["GAU", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["UUU", "UUC"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"],
    "I": ["AUU", "AUC", "AUA"],
    "K": ["AAA", "AAG"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "M": ["AUG"],
    "N": ["AAU", "AAC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
    "W": ["UGG"],
    "Y": ["UAU", "UAC"],
}


@click.command()
@click.option("--configuration-file", "-c", type=str, required=True, help="name of the config file")
@click.option(
    "--augmentation-type",
    "-type",
    type=str,
    required=True,
    help="dna / rna: specifies whether the output of the augmentation should be in DNA or RNA",
)
@click.option(
    "--augmentation-coeff",
    "-coeff",
    type=int,
    required=True,
    help="number of version to create per sequence",
)
@click.option(
    "--make-balance",
    "-balance",
    type=bool,
    default=False,
    help="Specifies whether or not the pos/neg ratio should be preserved after augmentation",
)
@click.option(
    "--keep-original",
    "-keep",
    type=bool,
    required=True,
    default=True,
    help="specifies whetehr or not to keep original sequences, or keep only newly generated ones",
)
@click.option("--dataset-path", "-data", type=str, required=True, help="Path to dataset file")
def augment(
    augmentation_type: str,
    keep_original: bool,
    configuration_file: str,
    dataset_path: str,
    augmentation_coeff: int,
    make_balance: bool,
) -> None:
    """Augments given dataframe. Augmented version will have a balanced positive / negative ratio."""
    assert augmentation_type in [
        "dna",
        "rna",
    ], "augmentation type should be one of the following: [dna, rna]"
    genetic_code = genetic_code_dna if augmentation_type == "dna" else genetic_code_rna

    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)

    augmented_wt_col_name = general_configuration["general_params"]["augmented_wildtype_col_name"]
    augmented_mutated_col_name = general_configuration["general_params"][
        "augmented_mutated_col_name"
    ]

    peptide_wt_col_name = general_configuration["general_params"]["peptide_wildtype_col_name"]
    peptide_mutated_col_name = general_configuration["general_params"]["peptide_mutated_col_name"]

    validation_set_proportion = general_configuration["general_params"]["validation_set_proportion"]
    label_name = general_configuration["general_params"]["label_name"]

    df = read_data(dataset_path)
    # (train / val) split and preserve (pos / neg) ratio
    pos_val_set = df[df[label_name] == 1].sample(frac=validation_set_proportion)
    neg_val_set = df[df[label_name] == 0].sample(frac=validation_set_proportion)
    val_set_ids = list(pos_val_set["id"].unique()) + list(neg_val_set["id"].unique())
    df["validation"] = 0
    df.loc[df["id"].isin(val_set_ids), "validation"] = 1

    pos_neg_ratio = len(df[df[label_name] == 1]) / len(df[df[label_name] == 0])
    wild_type_sequences = df[peptide_wt_col_name]
    mutated_sequences = df[peptide_mutated_col_name]
    labels = df[label_name]
    mutation_start_positions = [
        find_diff_pos(wt, mutated) for wt, mutated in zip(wild_type_sequences, mutated_sequences)
    ]

    pos_augmentation_coeff = augmentation_coeff
    if make_balance:
        neg_augmentation_coeff = int(augmentation_coeff * pos_neg_ratio)
    else:  # preserve initial pos / neg ratio
        neg_augmentation_coeff = augmentation_coeff

    augmentation_rows = []
    for idx in range(len(df)):
        log.info(f"sample: {idx}/{len(df)}")
        tmp_row = df.iloc[[idx]].copy()
        wt_peptide = wild_type_sequences[idx]
        mutated_peptide = mutated_sequences[idx]
        mutation_start_position = mutation_start_positions[idx]

        tmp_augmentation_coeff = int(labels[idx] * pos_augmentation_coeff) + int(
            (1 - labels[idx]) * neg_augmentation_coeff
        )

        for _ in tqdm(range(tmp_augmentation_coeff)):
            dna_wt = "".join([random.choice(genetic_code[base]) for base in wt_peptide])
            dna_mutated = "".join(
                [
                    dna_wt[: mutation_start_position * 3],
                    random.choice(genetic_code[mutated_peptide[mutation_start_position]]),
                    dna_wt[mutation_start_position * 3 + 3 :],
                ]
            )

            tmp_row[augmented_wt_col_name] = dna_wt
            tmp_row[augmented_mutated_col_name] = dna_mutated
            augmentation_rows.append(pd.DataFrame(tmp_row))

    if keep_original:
        augmentation_rows += [df]

    augmentation_df = pd.concat(augmentation_rows, ignore_index=True)
    augmentation_df = augmentation_df.sample(frac=1)  # shuffle
    file_name = dataset_path.split("/")[-1].split(".")[0]
    augmentation_df.to_csv(
        dataset_path.replace(file_name, f"augmented_{augmentation_type}_{file_name}")
    )
