"""Generate Dot product embeedings using ESM model from Meta. We generate the embeedings for mutated and wild type peptides."""


from typing import Any, List, Union

import click
import esm
import pandas as pd
import torch

from ig.src.utils import read_data


def remove_nans(dataset: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove nans from mutated and wild type peptides columns."""
    dataset = dataset[dataset[column].notna()]
    return dataset


def reset_index(dataset: pd.DataFrame) -> None:
    """Resetting the index fo the dataframe."""
    dataset.reset_index(inplace=True, drop=True)


def dot(list1: List, list2: List) -> float:
    """Dot product of two lists."""
    if len(list1) != len(list2):
        return 0

    return sum(i[0] * i[1] for i in zip(list1, list2))


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help=" Path to dataset containing peptide column.",
)
@click.option(
    "--column_mutated",
    "-cm",
    type=str,
    required=True,
    help=" Mutated peptide column name in the dataset.",
)
@click.option(
    "--column_wild",
    "-cw",
    type=str,
    required=True,
    help=" Wild type peptide column name in the dataset.",
)
@click.option(
    "--out",
    "-o",
    type=str,
    required=True,
    help=" name of the new dataset with mutation and wild type.",
)
@click.pass_context
def compute_embeddings(
    ctx: Union[click.core.Context, Any],  # pylint: disable=unused-argument,
    dataset: pd.DataFrame,
    column_mutated: str,
    column_wild: str,
    out: str,
) -> None:
    """Generate embeedings for mutated then wild type peptides and compute the dot product embeedings."""
    # Load data
    dataset = read_data(dataset)

    # remove nan values in column
    dataset = remove_nans(dataset=dataset, column=column_mutated)
    dataset = remove_nans(dataset=dataset, column=column_wild)

    # reset index of
    reset_index(dataset=dataset)

    # First phase: Calculate the embeddings for the mutated peptides
    list_of_tuples_mutated = []
    for i in range(len(dataset)):
        list_of_tuples_mutated.append(("mutated_peptides", dataset[column_mutated][i]))

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    _, _, batch_tokens = batch_converter(list_of_tuples_mutated)

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations_mutation = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations_mutation.append(
            token_representations[i, 1 : tokens_len - 1].mean(0, dtype=torch.float64).numpy()
        )
    # Incase you only want to include the embeddings for mutation
    # dataset['esm_embeddings_raw_mutation'] = pd.Series(sequence_representations_mutation, index=dataset.index)

    # First phase: Calculate the embeddings for the wild peptides
    list_of_tuples_wild = []
    for i in range(len(dataset)):
        list_of_tuples_wild.append(("wild_peptides", dataset[column_wild][i]))

    _, _, batch_tokens = batch_converter(list_of_tuples_wild)

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations_wild = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations_wild.append(
            token_representations[i, 1 : tokens_len - 1].mean(0, dtype=torch.float64).numpy()
        )
    # Incase you only want to include the embeddings for mutation
    # dataset['esm_embeddings_raw_wild'] = pd.Series(sequence_representations_wild, index=dataset.index)

    # Calculate the dot product of both embeddings
    dot_product = []
    for i, _ in enumerate(sequence_representations_mutation):
        dot_product.append(
            dot(sequence_representations_mutation[i], sequence_representations_wild[i])
        )

    dataset["esm_embeddings_dot_product_mutation_wild"] = pd.Series(
        dot_product, index=dataset.index
    )
    dataset.to_csv(out)
