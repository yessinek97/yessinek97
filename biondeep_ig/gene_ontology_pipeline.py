"""This module contains the pipeline that generates, reduces
and merges the Gene Ontology embeddings and Go term CC RNA representations to
the original dataset."""
import logging
import os
from pathlib import Path
from typing import Dict
from typing import List

import anc2vec
import click
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import decomposition
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# set the logger config
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def read_data(file_path: str) -> pd.DataFrame:
    """Read input data file.

    Args:
        file_path: Path of the input file.

    Returns:
        df: Input dataframe."""
    logging.info(f"Reading {file_path}")
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


def get_anc2vec_embeddings() -> Dict[str, np.ndarray]:
    """Get all the Go term embeddings dictionary."""
    return anc2vec.get_embeddings()


def calculate_sequence_embedding(sequence: str) -> np.ndarray:
    """Calculate go term embeddings for each sequence.

    This function takes the mean vector of the whole sequence
    vectors having shapes of (200,).

    Args:
        sequence: The input go terms sequence.

    Returns:
        sequence_embedding: The mean embedding for all the terms per sequence.

    """
    if not isinstance(sequence, str):
        return None
    # Split the Go terms
    go_sequence = sequence.split(";")
    embeddings = get_anc2vec_embeddings()
    sequence_vectors = []
    for go_term in go_sequence:
        try:
            sequence_vectors.append(embeddings[go_term])
            # Calculate the mean embedding from all the vectors
            sequence_embedding = np.mean(sequence_vectors, axis=0)

        except LookupError:
            pass
    return sequence_embedding


def generate_embeddings(data: pd.DataFrame, go_feature: str, embed_feature: str) -> pd.DataFrame:
    """Generate Gene Ontology embeddings for all the Go terms (cc, mf, bp).

    Adds the embedding features to the input dataset.

    Args:
        data: The input dataset.
        go_feature: The go term feature name.
        embed_feature: The go feature embedding name.
    """
    data[embed_feature] = data.apply(
        lambda row: calculate_sequence_embedding(row[go_feature]), axis=1
    )

    return data


def reduce_vectors(
    embeddings_df: pd.DataFrame,
    embedding_features: List[str],
    n_components: int = 3,
    technique: str = "pca",
):
    """This function allows applying the dimensionality reduction of a feature
    vector using a specific technique.

    Args:
        embeddings_df: The go term embeddings dataframe.
        embedding_features: The embedded go feature names.
        n_components: Number of components for the output reduced vector.
        technique: Dimensionality reduction technique ('pca','lsa','tsne').

    Returns:
        embeddings_df: The final dataset the reduced embeddings.
    """

    try:
        logging.info(
            f"Reducing the go term embeddings to {n_components} elements using {technique}"
        )

        for embedded_feature in embedding_features:

            embeddings = embeddings_df[embedded_feature].values
            embeddings = np.stack(embeddings, axis=0)
            # Standardize the go term embeddings
            embeddings_sc = StandardScaler().fit_transform(embeddings)

            if technique == "pca":

                pca = decomposition.PCA(n_components=n_components)

                pca_data = pca.fit_transform(embeddings_sc)

                # Create the reduced vectors
                for i in range(n_components):
                    embeddings_df[embedded_feature + f"_{i}"] = pca_data[:, i]
            elif technique == "lsa":
                embeddings_sc = csr_matrix(embeddings_sc)
                svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
                svd_data = svd.fit_transform(embeddings_sc)
                # Create the reduced vectors

                for i in range(n_components):

                    embeddings_df[embedded_feature + f"_{i}"] = svd_data[:, i]

            elif technique == "tsne":

                tsne = TSNE(
                    n_components=n_components, learning_rate="auto", init="random", perplexity=3
                )
                tsne_data = tsne.fit_transform(embeddings_sc)
                # Create the reduced vectors

                for i in range(n_components):

                    embeddings_df[embedded_feature + f"_{i}"] = tsne_data[:, i]
            else:
                logging.error(f"The specified {technique} is not allowed")

            logging.info(f"The {embedded_feature} vector has been reduced successfully!")

    except TypeError:
        pass

    return embeddings_df


def generate_rna_representation(sequence: str, indicator: str):
    """This function finds the rna localization element within
        a go term sequence.

    Args:
        sequence: The sequence including Go terms.
        indicator: a string indicator to map Go term CC RNA localization."""
    try:

        sequence_list = sequence.split(";")
        if indicator in sequence_list:

            representation = 1
        else:
            representation = 0
        return representation

    except AttributeError:
        pass


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
    "--output_path",
    "-o",
    type=str,
    required=True,
    help="Output Path to save the new dataset",
)
@click.option(
    "--n_components",
    "-c",
    type=int,
    required=True,
    help="Number of components for the embedding vectors to reduce",
)
@click.option(
    "--technique",
    "-t",
    type=str,
    required=True,
    help="The dimensionality reduction technique (pca,lsa,tsne)",
)
def main(data_path, go_terms, output_path, n_components=3, technique="pca"):
    """Run the script to generate the go embedding vectors.

    The Gene Ontology terms include cell compound (CC), molecular function(MF)
    and biological process(BP).

    data_path: path to the dataset containing the go features.
    go_terms: a list including the go terms included in the dataset.
    output_path: path to save the dataset with the go features embeddings.
    n_components: number of the components for dimensionality reduction.
    technique: dimensionality reduction technique.
    """
    data = read_data(data_path)
    embeddings_df = data.copy()
    # Split the go terms string input
    go_terms = go_terms.split()
    embeddings_df = embeddings_df[["id"] + go_terms]

    for go_feature in go_terms:
        # Generate the gene embeddings for each Go term
        logging.info(f"Generating the vector embeddings for {go_feature}")
        embeddings_df = generate_embeddings(embeddings_df, go_feature, go_feature + "_embed_vector")
        embeddings_df = embeddings_df.fillna(np.nan)
    embedded_features = [go_feature + "_embed_vector" for go_feature in go_terms]
    reduced_features = [
        embedded_feature + f"_{i}"
        for i in range(n_components)
        for embedded_feature in embedded_features
    ]

    # Drop the missing values for the embedded feature
    embeddings_df = embeddings_df.dropna(subset=embedded_features)
    reduced_df = reduce_vectors(embeddings_df, embedded_features, n_components, technique)
    reduced_df = reduced_df[reduced_features + ["id"]]
    # Merge the reduced features with the original dataset
    merged_data = pd.merge(data, reduced_df, on="id", how="left")
    # Go terms rna localization indicators
    go_rna_indicators = {
        "membrane": "GO:0016020",
        "nucleus": "GO:0005634",
        "endoplasmic": "GO:0005783",
        "ribosome": "GO:0005840",
        "cytosol": "GO:0005829",
        "exosome": "GO:00700062",
    }

    merged_data = merged_data.copy()
    # Make the rna_localization representation features for each sequence in go_term_cc for public data
    logging.info("Generating the Go term CC RNA representations")

    for rna_indicator, go_term in go_rna_indicators.items():

        merged_data[f"{rna_indicator}"] = merged_data["go_term_cc"].apply(
            lambda sequence: generate_rna_representation(sequence, go_term)
        )

    save_path = os.path.join(output_path)
    merged_data.to_csv(save_path)

    logging.info(f"The final dataset is saved to {save_path}")


if __name__ == "__main__":
    main()
