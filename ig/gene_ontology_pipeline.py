"""This module contains the implementation of the Gene Ontology pipeline."""
import os
import shutil
import time
from ast import literal_eval
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Union

import anc2vec
import click
import numpy as np
import pandas as pd
from google.cloud import storage
from scipy.sparse import csr_matrix
from sklearn import decomposition
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ig import CONFIGURATION_DIRECTORY
from ig.src.dataset import Dataset
from ig.src.logger import get_logger, init_logger
from ig.src.utils import load_yml, read_data, save_yml

log: Logger = get_logger("Gene Ontology Pipeline")


def check_file_exists(file_path: str, prefix: str = "gs://") -> bool:
    """This utility function checks whether the provided gs file exists or not.

    Args:
        file_path: file path.
        prefix: Google storage path prefix.

    Returns:
        exists: A boolean indicating the existence of the mentioned file in the GS bucket.
    """
    try:

        if file_path.startswith(prefix):

            storage_client = storage.Client()
            exists = storage.Blob.from_string(file_path).exists(storage_client)
        else:

            exists = os.path.exists(file_path)
    except AttributeError:
        exists = False
    return exists


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
            log.info("The specified %s is not allowed", technique)
        else:
            log.info(
                "Reducing the go term embeddings to %s elements using %s", n_components, technique
            )

        for embedded_feature in embedding_features:
            embeddings = embeddings_df[embedded_feature].values
            embeddings = np.stack(embeddings, axis=0)
            if check_file_exists(embeddings_path):
                log.info("Processing the ready-to-use GO embeddings for %s", embedded_feature)
                embeddings = process_vectors(embeddings_df, embedded_feature)

            # Standardize the go term embeddings
            embeddings_sc = StandardScaler().fit_transform(embeddings)

            if technique == "lsa":
                embeddings_sc = csr_matrix(embeddings_sc)

            dim_reduction = get_dimensionality_reduction(technique, n_components)
            reduced_data = dim_reduction.fit_transform(embeddings_sc)
            # Create the reduced vectors
            for i in range(n_components):
                embeddings_df[embedded_feature + f"_{i}"] = reduced_data[:, i]

            log.info("The %s vector has been reduced successfully!", embedded_feature)

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
    data[rna_indicator] = data["go_term_cc"].apply(
        lambda sequence: generate_rna_representation(sequence, go_term)
    )
    return data


def _check_data_experiment_folder(data_generation_dir_path: Path) -> None:
    """Check if the data generation folder exists or not to prepare the directory."""
    if data_generation_dir_path.exists():
        click.confirm(
            (
                f"A folder with the name {data_generation_dir_path.name} already exists."
                "Do you want to delete this directory and create a new one?"
                " Remember that all the files in this directory will be deleted!"
            ),
            abort=True,
        )
        shutil.rmtree(data_generation_dir_path)
    data_generation_dir_path.mkdir(exist_ok=True, parents=True)


def gene_ontology_pipeline_per_binder(
    ctx: Union[click.core.Context, Any],
    data_config: Dict[str, Any],
    go_embeddings_config: Dict[str, Any],
    go_terms: str,
    output_path: Path,
    binder: str,
) -> int:
    """This function generates gene ontology features per binder.

    Args:
        ctx: context object.
        data_config: a dictionary holding data configuration.
        go_embeddings_config: a dictionary holding go_embeddings configuration.
        go_terms: a string including the go terms used in the dataset.
        output_path: path to save the dataset with the go features embeddings.
        binder: dataset binder (netmhcpan, biondeep).
    """
    data_version = data_config[binder]["version"]
    data_files = data_config[binder]["paths"]

    log.info(
        "Working on %s binder version %s with %s available data files",
        binder,
        data_version,
        len(data_files),
    )

    technique = go_embeddings_config["dimensionality_reduction"]["technique"]
    n_components = go_embeddings_config["dimensionality_reduction"]["n_components"]
    save_embeddings = go_embeddings_config["save_embeddings"]
    go_embeddings_version = go_embeddings_config["version"]
    output_path = Path(output_path)
    binder_dir_path = output_path / f"{binder}_{data_version}"

    _check_data_experiment_folder(binder_dir_path)

    embeddings_save_dir_path = binder_dir_path / "embeddings"
    generated_datasets_dir_path = binder_dir_path / "generated"
    configuration_files_path = binder_dir_path / "configuration"
    binder_dir_path.mkdir(exist_ok=True, parents=True)
    embeddings_save_dir_path.mkdir(exist_ok=True, parents=True)
    generated_datasets_dir_path.mkdir(exist_ok=True, parents=True)

    experiment_path = Path(configuration_files_path)
    experiment_path.mkdir(exist_ok=True, parents=True)

    data_loader = Dataset(
        click_ctx=ctx,
        data_path=data_config[binder]["paths"]["public"],
        configuration=data_config[binder],
        is_train=True,
        experiment_path=experiment_path,
        force_gcp=True,
    )
    embedding_paths = go_embeddings_config["embedding_paths"]
    total_elapsed_time = 0

    for data_name, data_path in data_files.items():
        log.info("Processing the %s dataset", data_name)
        start = time.time()
        if not check_file_exists(data_path):

            log.info("The provided file at %s does not exist", data_path)

        data = read_data(data_path, low_memory=False)

        data = data.rename(columns={"ID": "id", "Id": "id"})
        embeddings_df = data.copy()
        # Split the go terms string input
        go_sequence = go_terms.split()
        embeddings_df = embeddings_df[["id"] + go_sequence]
        if embedding_paths:

            embedding_path = embedding_paths[data_name]

        if check_file_exists(embedding_path):
            # Using custom Go embeddings
            log.info("The provided file at %s exists", embedding_path)

            log.info("Loading the GO Anc2vec embeddings from %s", embedding_path)
            embeddings_df = pd.read_csv(embedding_path, low_memory=False)

        else:
            log.warning("The Anc2vec embeddings are not found !")
            log.info("Generating the Gene Ontology vector embeddings from scratch")

            # Generate Gene Ontology embeddings if they don't exist
            for go_feature in go_sequence:
                # Generate the gene embeddings for each Go term
                log.info("Generating the vector embeddings for %s", go_feature)
                embeddings_df = generate_embeddings(
                    embeddings_df, go_feature, go_feature + "_embed_vector"
                )
                embeddings_df = embeddings_df.fillna(np.nan)
        if save_embeddings:
            embeddings_output_path = (
                embeddings_save_dir_path / f"{data_name}_go_embeddings_{go_embeddings_version}.csv"
            )

            embeddings_df.to_csv(embeddings_output_path, index=False)
            log.info(
                "Saved the Go Anc2vec embeddings version %s  in %s",
                go_embeddings_version,
                embeddings_output_path,
            )

        embedded_features = [go_feature + "_embed_vector" for go_feature in go_sequence]
        reduced_features = [
            embedded_feature + f"_{i}"
            for i in range(n_components)
            for embedded_feature in embedded_features
        ]
        # Drop the missing values for the embedded feature
        embeddings_df = embeddings_df.dropna(subset=embedded_features)

        reduced_df = reduce_vectors(
            embeddings_df, embedded_features, embedding_path, n_components, technique
        )
        reduced_df = reduced_df[
            reduced_features
            + [
                "id",
            ]
        ].reset_index(drop=True)

        # Merge the reduced features with the original dataset
        merged_data = pd.merge(data, reduced_df, on="id", how="left")
        # Go terms rna localization indicators
        go_rna_indicators = {
            "go_cc_rna_loc_membrane": "GO:0016020",
            "go_cc_rna_loc_nucleus": "GO:0005634",
            "go_cc_rna_loc_endoplasmic": "GO:0005783",
            "go_cc_rna_loc_ribosome": "GO:0005840",
            "go_cc_rna_loc_cytosol": "GO:0005829",
            "go_cc_rna_loc_exosome": "GO:00700062",
        }
        go_features = reduced_features + list(go_rna_indicators.keys())
        # Find the old occurences of Gene ontology features in the configuration file
        old_go_features = data_loader.find_features(
            data_loader.features_configuration["float"], "go_"
        )
        # Delete the old occurences of Gene Ontology features if there are any and
        # replace them with the new ones
        data_loader.features_configuration["float"] = sorted(
            set(data_loader.features_configuration["float"] + go_features) - set(old_go_features)
        )

        features_configuration_file_name = Path(
            data_config[binder]["processing"]["trainable_features"]
        ).name
        file_name, extension = features_configuration_file_name.split(".")

        features_configuration_file_name = file_name + "_go." + extension

        features_configuration_save_path = (
            configuration_files_path / features_configuration_file_name
        )

        log.info(
            "Saving the updated features configuration file under %s",
            features_configuration_save_path,
        )
        save_yml(data_loader.features_configuration, features_configuration_save_path)
        merged_data = merged_data.copy()
        # Make the rna_localization representation features for each sequence in go_term_cc for public data
        log.info("Generating the Go term CC RNA representations")

        for rna_indicator, go_term in go_rna_indicators.items():
            merged_data = apply_generation(merged_data, rna_indicator, go_term)

        log.info("The Go term CC RNA representations have been generated successfully!")

        save_path = generated_datasets_dir_path / f"{data_name}.go.csv"
        merged_data.to_csv(save_path, index=False)
        elapsed_time = int(time.time() - start)
        total_elapsed_time += elapsed_time
        log.info("The final dataset is saved to %s", save_path)
        log.info(
            "The elapsed time to generate the %s dataset is %s seconds", data_name, elapsed_time
        )
    log.info(
        "The total elapsed time to generate all the datasets for %s binder is %s seconds",
        binder,
        total_elapsed_time,
    )
    return total_elapsed_time


@click.command()
@click.option(
    "--configuration_path",
    "-c",
    type=str,
    required=True,
    help="Path to the configuration file",
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
@click.pass_context
def gene_ontology_pipeline(
    ctx: Union[click.core.Context, Any],
    configuration_path: Path,
    go_terms: str,
    output_path: Path,
) -> None:
    """Run the script to generate the go embedding vectors.

    The Gene Ontology terms include cell compound (CC), molecular function(MF)
    and biological process(BP).
    configuration_path: path to the gene_ontology configuration file.
    go_terms: a string including the go terms used in the dataset.
    output_path: path to save the dataset with the go features embeddings.
    """
    init_logger(Path(output_path), "Go_data_generation")

    if configuration_path:
        configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_path)
    data_config = configuration["dataset"]
    go_embeddings_config = configuration["go_features"]
    binders = list(data_config.keys())
    final_elapsed_time = 0

    for binder in binders:
        elapsed_time_per_binder = gene_ontology_pipeline_per_binder(
            ctx, data_config, go_embeddings_config, go_terms, output_path, binder
        )
        final_elapsed_time += elapsed_time_per_binder

    log.info(
        "The total elapsed time to generate all the datasets for all binders is %s seconds",
        final_elapsed_time,
    )
