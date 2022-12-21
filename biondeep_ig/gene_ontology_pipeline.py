"""This module contains the implementation of the Gene Ontology pipeline."""
import logging
import os
from ast import literal_eval
from pathlib import Path
from typing import Any
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
    """This function is used to read the input dataset.

    Args:
        file_path: Path of the input dataset.

    Returns:
        df: The input dataframe.
    """
    logging.info(f"Loading {file_path}")
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


def process_vectors(data: pd.DataFrame, embedding_feature: str) -> np.ndarray:
    """This function is used to clean the embedded feature vectors.

    Args:
        data: The embeddings data.
        embedding_feature: The Gene Ontology embedding to clean.

    Returns:
        vectors: The clean embeddings vectors.
    """
    try:

        vectors = data[embedding_feature]

        vectors = vectors.str.strip("[]").replace("\n", "").str.split()
        vectors = vectors.apply(lambda x: np.array([literal_eval(i) for i in x]))
        vectors = np.stack(vectors, axis=0)
    except TypeError:
        pass

    return vectors


def get_dimensionality_reduction(technique: str, n_components: int) -> Any:
    """This helper function prepares the specified dimensionality technique object with an input arguments.

    Args:
        technique: Dimensionality reduction technique ('pca','lsa','tsne').
        n_components: Number of components for the output reduced vector.

    Returns:
        dim_reduction_object: The dimensionality reduction object.
    """
    dim_reduction = {
        "pca": decomposition.PCA(n_components=n_components),
        "lsa": TruncatedSVD(n_components=n_components, n_iter=7, random_state=42),
        "tsne": TSNE(
            n_components=n_components,
            learning_rate="auto",
            init="random",
            perplexity=3,
            random_state=42,
        ),
    }
    dim_reduction_object = dim_reduction[technique]

    return dim_reduction_object


def reduce_vectors(
    embeddings_df: pd.DataFrame,
    embedding_features: List[str],
    embeddings_path: str,
    n_components: int = 3,
    technique: str = "pca",
) -> pd.DataFrame:
    """This function allows applying the dimensionality reduction of a feature vector using a specific technique.

    Args:
        embeddings_df: The go term embeddings dataframe.
        embedding_features: The embedded go feature names.
        n_components: Number of components for the output reduced vector.
        technique: Dimensionality reduction technique ('pca','lsa','tsne').
        embeddings_path: Path of ready-to-use generated Go embeddings.

    Returns:
        embeddings_df: The final dataset the reduced embeddings.
    """
    try:

        if technique not in ["pca", "lsa", "tsne"]:
            logging.error(f"The specified {technique} is not allowed")
        else:
            logging.info(
                f"Reducing the go term embeddings to {n_components} elements using {technique}"
            )

        for embedded_feature in embedding_features:

            if os.path.exists(embeddings_path):
                logging.info(f"Processing the ready-to-use GO embeddings for {embedded_feature}")
                embeddings = process_vectors(embeddings_df, embedded_feature)
            else:
                embeddings = embeddings_df[embedded_feature].values
                embeddings = np.stack(embeddings, axis=0)

            # Standardize the go term embeddings
            embeddings_sc = StandardScaler().fit_transform(embeddings)

            if technique == "lsa":
                embeddings_sc = csr_matrix(embeddings_sc)

            dim_reduction = get_dimensionality_reduction(technique, n_components)
            reduced_data = dim_reduction.fit_transform(embeddings_sc)
            # Create the reduced vectors
            for i in range(n_components):
                embeddings_df[embedded_feature + f"_{i}"] = reduced_data[:, i]

            logging.info(f"The {embedded_feature} vector has been reduced successfully!")

    except TypeError:
        pass

    return embeddings_df


def generate_rna_representation(sequence: str, indicator: str) -> int:
    """This function finds the rna localization element within a go term sequence.

    Args:
        sequence: The sequence including Go terms.
        indicator: a string indicator to map Go term CC RNA localization.

    Returns:
        representation: The Go term CC RNA representations.
    """
    try:

        sequence_list = sequence.split(";")
        representation = 0
        if indicator in sequence_list:

            representation = 1
        else:
            representation = 0

    except AttributeError:
        representation = np.nan

    return representation


def apply_generation(data: pd.DataFrame, rna_indicator: str, go_term: str) -> pd.DataFrame:
    """This helper function applies the generation of Go term CC RNA representations to all the sequences.

    Args:
        data: The input data.
        rna_indicator: The rna indicator to index with.
        go_term: The specified Go term CC to map.

    Returns:
        data: The output dataset.
    """
    data[f"{rna_indicator}"] = data["go_term_cc"].apply(
        lambda sequence: generate_rna_representation(sequence, go_term)
    )
    return data


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
    help="Go terms column names separated by a space.",
)
@click.option(
    "--output_path",
    "-o",
    type=str,
    required=True,
    help="Output Path to save the new dataset",
)
@click.option(
    "--embeddings_path",
    "-e",
    type=str,
    default="",
    required=False,
    help="Input path of existing Gene Ontology embeddings",
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
@click.option(
    "--save_embeddings",
    "-s",
    is_flag=True,
    help="Save the generated Gene Ontology embedding vectors (200 elements)",
)
def main(
    data_path: str,
    go_terms: str,
    output_path: str,
    embeddings_path: str,
    n_components: int = 3,
    technique: str = "pca",
    save_embeddings=False,
) -> None:
    """Run the script to generate the go embedding vectors.

    The Gene Ontology terms include cell compound (CC), molecular function(MF)
    and biological process(BP).

    data_path: path to the dataset containing the go features.
    go_terms: a string including the go terms used in the dataset.
    output_path: path to save the dataset with the go features embeddings.
    n_components: number of the components for dimensionality reduction.
    technique: dimensionality reduction technique.
    embeddings_path: Path of ready-to-use generated Go embeddings.
    save_embeddings: A boolean to control saving the embeddings only.
    """
    data = read_data(data_path)
    embeddings_df = data.copy()
    # Split the go terms string input
    go_sequence = go_terms.split()
    embeddings_df = embeddings_df[["id"] + go_sequence]
    if os.path.exists(embeddings_path):
        # Using the already generated Go embeddings
        logging.info(f"Loading the ready-to-use GO feature from {embeddings_path}")
        embeddings_df = pd.read_csv(embeddings_path)

    else:
        # Generate Gene Ontology embeddings if they don't exist
        for go_feature in go_sequence:
            # Generate the gene embeddings for each Go term
            logging.info(f"Generating the vector embeddings for {go_feature}")
            embeddings_df = generate_embeddings(
                embeddings_df, go_feature, go_feature + "_embed_vector"
            )
            embeddings_df = embeddings_df.fillna(np.nan)
    if save_embeddings:
        embeddings_output_path = (
            os.path.dirname(data_path) + "/go_embeddings_" + os.path.basename(data_path)
        )
        embeddings_df.to_csv(embeddings_output_path, index=False)
        logging.info(f"Saved the Gene Ontology features in {embeddings_output_path}")

    embedded_features = [go_feature + "_embed_vector" for go_feature in go_sequence]
    reduced_features = [
        embedded_feature + f"_{i}"
        for i in range(n_components)
        for embedded_feature in embedded_features
    ]

    # Drop the missing values for the embedded feature
    embeddings_df = embeddings_df.dropna(subset=embedded_features)

    reduced_df = reduce_vectors(
        embeddings_df, embedded_features, embeddings_path, n_components, technique
    )
    reduced_df = reduced_df[
        reduced_features
        + ["id", "go_term_cc_embed_vector", "go_term_bp_embed_vector", "go_term_mf_embed_vector"]
    ]

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
        merged_data = apply_generation(merged_data, rna_indicator, go_term)

    logging.info("The Go term CC RNA representations have been generated successfully!")

    save_path = os.path.join(output_path)
    merged_data.to_csv(save_path, index=False)
    logging.info(f"The final dataset is saved to {save_path}")


if __name__ == "__main__":
    main()
