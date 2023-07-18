# type: ignore
"""This module includes the test functions for the gene ontology pipeline."""
import os
from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from ig.gene_ontology_pipeline import (
    apply_generation,
    calculate_sequence_embedding,
    gene_ontology_pipeline,
    generate_embeddings,
    generate_rna_representation,
    get_anc2vec_embeddings,
    process_vectors,
    read_data,
    reduce_vectors,
)
from ig.src.utils import load_yml, save_yml


def test_read_data(file_paths: List[str]) -> None:
    """This function tests the read_data helper function."""
    for file_path in file_paths:
        expected_data = read_data(file_path)
        assert isinstance(
            expected_data, pd.DataFrame
        ), "Something wrong with reading the input dataset!"


def test_get_anc2vec_embeddings() -> None:
    """This function tests the get_anc2vec_embeddings helper function."""
    embeddings_dict = get_anc2vec_embeddings()

    assert isinstance(embeddings_dict, Dict), "Check the anc2vec embeddings!"


def test_calculate_sequence_embedding(sequence: str) -> None:
    """This function tests calculate_sequence_embedding function."""
    embedding = calculate_sequence_embedding(sequence)
    assert embedding.shape == (200,), "Check the embedding vector !"
    assert float(np.mean(embedding)) == -0.009362175129354, "Check the embedding vector values!"
    assert float(np.max(embedding)) == 0.3775474429130554, "Check the embedding vector values!"
    assert float(np.min(embedding)) == -0.37461772561073303, "Check the embedding vector values!"
    assert (
        float(np.median(embedding)) == -0.010463468730449677
    ), "Check the embedding vector values!"


def test_generate_embeddings(
    test_df: pd.DataFrame,
    go_term: str = "go_term_cc",
    embedded_feature: str = "go_term_cc_embed_vector",
) -> None:
    """This function tests generating the embeddings for all the sequences as implemented in generate_embeddings function."""
    test_df = test_df.iloc[:3]
    data = generate_embeddings(test_df, go_term, embedded_feature)

    assert embedded_feature in data.columns
    assert data[embedded_feature].shape == (3,)
    assert (
        float(np.mean(np.mean(data[embedded_feature], axis=0))) == 0.0011103653814643621
    ), "Something wrong with the embeddings generation"


def test_process_vectors(
    test_go_embeddings_df: pd.DataFrame, embedded_feature: str = "go_term_cc_embed_vector"
) -> None:
    """This function tests the process_vectors function."""
    processed_vectors = process_vectors(test_go_embeddings_df, embedded_feature)
    assert processed_vectors.shape == (3, 200), "Check the processed vectors !"
    assert (
        float(np.mean(np.mean(processed_vectors, axis=0))) == 0.0011103643333333336
    ), "Something wrong with the processed vectors"


def test_reduce_vectors(
    test_go_embeddings_df: pd.DataFrame,
    embedding_features: List[str],
    n_components: int,
    techniques: List[str],
    embeddings_path: str = "tests/fixtures/test_data_go_embeddings.csv",
) -> None:
    """This function includes the tests for reduce_vectors function."""
    reduced_df = reduce_vectors(
        test_go_embeddings_df, embedding_features, embeddings_path, n_components, techniques[0]
    )
    reduced_features = [
        embedding_feature + f"_{i}"
        for i in range(n_components)
        for embedding_feature in embedding_features
    ]
    assert np.array_equal(
        reduced_df[reduced_features].values.astype("float32"),
        np.array(
            [
                [-1.0000000e01, 5.9708547e-15, 0.0000000e00],
                [-1.0000000e01, -1.1941709e-15, 1.9870841e-30],
                [2.0000000e01, 2.3883417e-15, 9.9354203e-31],
            ],
            dtype="float32",
        ),
    ), "Check the reduced vectors !"


def test_generate_rna_representation(sequence: str, indicator: str = "GO:0005634") -> None:
    """This function tests generate_rna_representation function."""
    rna_representation = generate_rna_representation(sequence, indicator)
    assert rna_representation == 1, "Check the Go term CC RNA representation fuction"


def test_apply_generation(
    test_df: pd.DataFrame, rna_indicator: str = "membrane", go_term: str = "GO:0016020"
) -> None:
    """This function tests apply_generation function for Go term CC RNA representation."""
    test_df = test_df.iloc[:3]
    test_df_rna_representation = apply_generation(test_df, rna_indicator, go_term)
    assert np.array_equal(test_df_rna_representation[rna_indicator].values, np.array([1, 1, 1]))


@mock.patch("click.confirm")
@pytest.mark.parametrize("embeddings_path", ["", "tests/fixtures/public_go_embeddings_V1.csv"])
def test_gene_ontology_pipeline(
    mock_click: mock.MagicMock,
    embeddings_path: str,
    gene_ontology_config_path: str,
    input_go_terms: str,
    output_path: str = "tests/fixtures/go_pipeline_test",
) -> None:
    """This function will test the whole gene ontology pipeline."""
    mock_click.return_value = "y"
    configuration_file = load_yml("tests/configuration/gene_ontology.yml")
    configuration_file["go_features"]["embedding_paths"]["public"] = embeddings_path
    save_yml(configuration_file, "tests/configuration/gene_ontology.yml")
    runner = CliRunner()
    params = [
        "--configuration_path",
        gene_ontology_config_path,
        "--go_terms",
        input_go_terms,
        "--output_path",
        output_path,
    ]

    _ = runner.invoke(gene_ontology_pipeline, params)

    generated_dataset_path = os.path.join(output_path, "dummy_V1/generated/public.go.csv")
    generated_embeddings_path = os.path.join(
        output_path, "dummy_V1/embeddings/public_go_embeddings_V1.csv"
    )
    generated_configuration_file_path = os.path.join(
        output_path, "dummy_V1/configuration/features_quickstart_go.yml"
    )
    embedding_vectors = [
        "go_term_cc_embed_vector",
        "go_term_bp_embed_vector",
        "go_term_mf_embed_vector",
    ]
    reduced_vectors = [
        "go_term_bp_embed_vector_0",
        "go_term_bp_embed_vector_1",
        "go_term_bp_embed_vector_2",
        "go_term_cc_embed_vector_0",
        "go_term_cc_embed_vector_1",
        "go_term_cc_embed_vector_2",
        "go_term_mf_embed_vector_0",
        "go_term_mf_embed_vector_1",
        "go_term_mf_embed_vector_2",
    ]
    go_cc_rna_loc_features = [
        "go_cc_rna_loc_membrane",
        "go_cc_rna_loc_nucleus",
        "go_cc_rna_loc_endoplasmic",
        "go_cc_rna_loc_ribosome",
        "go_cc_rna_loc_cytosol",
        "go_cc_rna_loc_exosome",
    ]
    generated_features = reduced_vectors + go_cc_rna_loc_features
    assert os.path.exists(generated_dataset_path), "Check the Gene ontology main pipeline !"
    assert set(pd.read_csv(generated_dataset_path).columns).intersection(generated_features) == set(
        generated_features
    ), "Check the generated Gene ontology dataset!"
    assert set(pd.read_csv(generated_embeddings_path).columns).intersection(
        embedding_vectors
    ) == set(embedding_vectors), "Check the generated embeddings!"
    assert set(load_yml(generated_configuration_file_path)["float"]).intersection(
        generated_features
    ) == set(generated_features), "Check the generated configuration file!"
