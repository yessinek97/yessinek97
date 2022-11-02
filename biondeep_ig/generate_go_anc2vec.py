"""Generate the Gene ontology features using the Anc2vec package."""
import os
from pathlib import Path

import anc2vec
import click
import numpy as np
import pandas as pd


def read_data(file_path: str):
    """Read data."""
    extension = Path(file_path).suffix
    if extension == ".csv":
        df = pd.read_csv(file_path)

    elif extension == ".tsv":
        df = pd.read_csv(file_path, sep="\t")

    elif extension == ".xlsx":
        df = pd.read_excel(file_path)

    else:
        raise ValueError(f"extension {extension} not supported")

    return df


def get_embeddings():
    """Generate go embedding with anc2vec."""
    return anc2vec.get_embeddings()


def calculate_mean_go_terms(terms):
    """Calculate the mean of the go terms for a row."""
    if not isinstance(terms, str):
        return None
    list_terms = terms.split(";")
    embeds = get_embeddings()
    means = []
    for item in list_terms:
        try:
            means.append(embeds[item].mean())
        except LookupError:
            pass
    return np.mean(means)


def apply_function(data, term, term_embedd):
    """Apply the go calculation embedding on a column."""
    data[term_embedd] = data.apply(lambda row: calculate_mean_go_terms(row[term]), axis=1)


@click.command()
@click.option(
    "--data_path",
    "-data",
    type=str,
    required=True,
    help="Path to the dataset you want to add the go feature to.",
)
@click.option(
    "--go_terms",
    "-go",
    type=str,
    required=True,
    help="Go terms seperated by a space.",
)
@click.option(
    "--output_csv",
    "-o",
    type=str,
    required=True,
    help="Path to the dataset you want to add the go feature to.",
)
def main(data_path, go_terms, output_csv):
    """Run the script to generate.

    embeddings for all the columns provided.

    data_path: path to the dataset containg the go features.
    go_terms: columns of the go features in the dataset.
    output_path: path to the next dataset with the go features embedding.
    """
    data = read_data(data_path)
    go_term_csv = data["ID"]
    for term in go_terms.split():
        apply_function(data, term, term + "embed")
        go_term_csv[term + "embed"] = data[term + "embed"]
    data.to_csv(output_csv)
    go_term_csv.to_csv(os.path.splitext(os.path.basename(data_path))[0] + "go_terms_alone.csv")
    print("SUCESS")


if __name__ == "__main__":
    main()
