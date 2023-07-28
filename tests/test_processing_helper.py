"""This module includes unit tests for the processing helper module."""

import os
from pathlib import Path
from typing import Any, Dict, List, Set, Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ig import CONFIGURATION_DIRECTORY
from ig.src.processing_helper import (
    add_features,
    clean_target,
    cross_validation,
    data_description,
    data_processor_new_data,
    data_processor_single_data,
    find_duplicated_columns_by_values,
    get_columns_by_keywords,
    get_data_type,
    get_duplicate_columns,
    get_features_from_features_configuration,
    get_features_type,
    get_list_type,
    is_bool,
    is_integer,
    is_list,
    legend_replace_renamed_columns,
    load_dataset,
    preprocess_train,
    process_expression,
    remove_columns,
    remove_columns_with_unique_value,
    remove_duplicate_columns,
    remove_duplicated_columns_by_name,
    remove_list_columns,
    remove_nan_columns,
    remove_rows,
    rename_scores,
    replace_nan_strings,
    report_missing_columns,
    sort_duplicated_columns,
    track_features,
    union_type_from_features_type,
)


def test_load_dataset(helper_dummy_data_path: str, helper_dummy_df: pd.DataFrame) -> None:
    """This function is used to test the behavior of load_dataset function."""
    data = load_dataset(helper_dummy_data_path, keep_default_na=False)

    assert data.equals(helper_dummy_df), "Check processing helper functions: load_dataset!"


def test_replace_nan_strings(helper_dummy_df: pd.DataFrame) -> None:
    """This function is used to test the behavior of replace_nan_strings function."""
    data = replace_nan_strings(helper_dummy_df)
    assert data.isna().sum().sum() == 3, "Check processing helper functions: replace_nan_strings!"


def test_remove_rows() -> None:
    """This function is used to test the behavior of remove_rows function."""
    data = pd.DataFrame({"col1": ["A", "B", "C"], "col2": ["X", "Y", "Z"]})
    configuration = MagicMock(filter_rows_column="col1", filter_rows_value="B")
    expected_data = pd.DataFrame({"col1": ["A", "C"], "col2": ["X", "Z"]}, index=[0, 2])

    assert remove_rows(data, configuration).equals(
        expected_data
    ), "Check processing helper functions: remove_rows!"


def test_clean_target() -> None:
    """This function is used to test the behavior of clean_target function."""
    data = pd.DataFrame(
        {
            "col1": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "target": [1, "2", "NA", "NAN", np.nan, 0, "1", "0"],
        }
    )
    expected_data = pd.DataFrame(
        {"col1": ["A", "F", "G", "H"], "target": [1, 0, 1, 0]}, index=[0, 5, 6, 7]
    )

    assert clean_target(data, "target").equals(
        expected_data
    ), "Check processing helper functions: clean_target!"


def test_data_description() -> None:
    """This function is used to test the behavior of data_description function."""
    data = pd.DataFrame({"A": [1, 1, np.nan, np.nan], "B": [12, 4, 5, 3], "C": [7.2, 8, 9.4, 0]})
    names = ["A", "B", "C"]
    expected_output = pd.DataFrame(
        {
            "Name": ["A", "B", "C"],
            "Type": [np.dtype("float64"), np.dtype("int64"), np.dtype("float64")],
            "Min": [1.0, 3.0, 0.0],
            "Max": [1.0, 12.0, 9.4],
            "Mean": [1.0, 6.0, 6.15],
            "Nan count": [2, 0, 0],
            "Nan Ratio": [0.5, 0.0, 0.0],
            "Nunique": [1, 4, 4],
            "Unique": [[1.0, np.nan], [12, 4, 5, 3], [7.2, 8.0, 9.4, 0.0]],
        }
    )

    result = data_description(data, names)

    pd.testing.assert_frame_equal(
        result, expected_output, obj="Check processing helper functions: data_description!"
    )


def test_remove_columns() -> None:
    """This function is used to test the behavior of remove_columns function."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    expected_data = pd.DataFrame({"col1": [1, 2, 3], "col3": [7, 8, 9]})
    assert remove_columns(data, ["col2"]).equals(
        expected_data
    ), "Check processing helper functions: remove_columns!"


def test_remove_nan_columns() -> None:
    """This function is used to test the behavior of remove_nan_columns function."""
    data = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [4, np.nan, np.nan], "col3": [pd.NA, pd.NA, pd.NA]}
    )
    configuration = MagicMock(nan_ratio=0.5, keep_features=False, include_features=["col1"])
    expected_data = pd.DataFrame({"col1": [1, 2, 3]})
    assert remove_nan_columns(
        data, configuration.nan_ratio, configuration.keep_features, configuration.include_features
    ).equals(expected_data), "Check processing helper functions: remove_nan_columns!"


def test_remove_duplicated_columns_by_name() -> None:
    """This function is used to test the behavior of remove_duplicated_columns_by_name function."""
    data = pd.DataFrame([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13]])
    data.columns = ["col1", "col2", "col3", "col4"]
    data.rename(columns={"col4": "col2"}, inplace=True)

    expected_data = pd.DataFrame({"col1": [1, 4, 7, 10], "col3": [3, 6, 9, 12]})

    assert remove_duplicated_columns_by_name(data).equals(
        expected_data
    ), "Check processing helper functions: remove_duplicated_columns_by_name!"


def test_remove_columns_with_unique_value() -> None:
    """This function is used to test the behavior of remove_columns_with_unique_value function."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 7, 7]})
    expected_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    assert remove_columns_with_unique_value(data).equals(
        expected_data
    ), "Check processing helper functions: remove_columns_with_unique_value!"


@pytest.mark.parametrize(
    ("input_str", "expected_result"),
    [
        ("[1, 2, 3]", 1),  # Valid list
        ("[1, 2, 3", 0),  # Missing closing bracket
        ("1, 2, 3]", 0),  # Missing opening bracket
        ("1, 2, 3", 0),  # No brackets
        ("", 0),  # Empty string
        ("[]", 1),  # Empty list
        ("[a, b, c]", 1),  # List with strings
        ("[1, [2, 3], 4]", 1),  # Nested list
    ],
)
def test_is_list(input_str: str, expected_result: int) -> None:
    """This function is used to test the behavior of is_list function."""
    assert is_list(input_str) == expected_result, "Check processing helper functions: !"


def test_remove_list_columns() -> None:
    """This function is used to test the behavior of remove_list_columns function."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": "[[4], [5], [6]]", "col3": "[[7], [7], [7]]"})
    expected_data = pd.DataFrame({"col1": [1, 2, 3]})
    assert remove_list_columns(data).equals(
        expected_data
    ), "Check processing helper functions: remove_list_columns!"


@pytest.mark.parametrize(
    ("input_data", "scores", "expected_output"),
    [
        # Test case 1: Some columns renamed
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
            {"A": "Score1", "C": "Score2"},
            pd.DataFrame({"Score1": [1, 2, 3], "B": [4, 5, 6], "Score2": [7, 8, 9]}),
        ),
        # Test case 2: All columns renamed
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
            {"A": "Score1", "B": "Score2", "C": "Score3"},
            pd.DataFrame({"Score1": [1, 2, 3], "Score2": [4, 5, 6], "Score3": [7, 8, 9]}),
        ),
        # Test case 3: Empty DataFrame
        (pd.DataFrame(), {"A": "Score1", "B": "Score2"}, pd.DataFrame()),
        # Test case 4: Empty scores dictionary
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
            {},
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
        ),
    ],
)
def test_rename_scores(
    input_data: pd.DataFrame, scores: Dict[str, str], expected_output: pd.DataFrame
) -> None:
    """This function is used to test the behavior of rename_scores function."""
    assert rename_scores(input_data, scores).equals(
        expected_output
    ), "Check processing helper functions: rename_scores!"


@pytest.mark.parametrize(
    ("columns_list", "keywords", "expected_result"),
    [
        # Test case 1: Single keyword as a string
        (["name", "age", "gender", "address"], "name", ["name"]),
        # Test case 2: Multiple keywords as a list
        (["name", "age", "gender", "address"], ["name", "age"], ["name", "age"]),
        # Test case 3: Empty columns_list
        ([], ["name", "age"], []),
        # Test case 4: Empty keywords list
        (["name", "age", "gender", "address"], [], []),
        # Test case 5: No matching columns
        (["name", "age", "gender", "address"], "city", []),
        # Test case 6 Regular expression keyword
        (
            ["first_name", "last_name", "age", "address"],
            r"^[a-z]+_name$",
            ["first_name", "last_name"],
        ),
    ],
)
def test_get_columns_by_keywords(
    columns_list: List[str], keywords: List[str], expected_result: List[str]
) -> None:
    """This function is used to test the behavior of get_columns_by_keywords function."""
    assert (
        get_columns_by_keywords(columns_list, keywords) == expected_result
    ), "Check processing helper functions: get_columns_by_keywords!"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10, True),
        (3.14, False),
        ("123", False),
    ],
)
def test_is_integer(value: int, expected: bool) -> None:
    """This function is used to test the behavior of is_integer function."""
    assert is_integer(value) == expected, "Check processing helper functions: is_integer!"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, True),
        (0, False),
        (1, False),
    ],
)
def test_is_bool(value: Union[bool, int], expected: bool) -> None:
    """This function is used to test the behavior of is_bool function."""
    assert is_bool(value) == expected, "Check processing helper functions: is_bool!"


def test_process_expression(expression_dummy_data: pd.DataFrame) -> None:
    """This function is used to test the behavior of process_expression function."""
    configuration = MagicMock(
        expression_raw_name="gene_expression_1",
        expression_name="expression",
        include_features=["rnalocalization", "gene_ontology"],
        expression_filter="exp",
    )
    # Process the expression columns
    processed_data = process_expression(
        expression_dummy_data, configuration, configuration.include_features
    )

    # Check if the expression column is added to the data and columns from include_features in are in the data
    assert (
        "expression" in processed_data.columns
    ), "Check processing helper functions: process_expression !"
    assert (
        "rnalocalization" in processed_data.columns
    ), "Check processing helper functions: process_expression !"
    # Check if the original expression column is removed and columns from include_features that are not in the data are ignored
    assert (
        "gene_expression_1" not in processed_data.columns
    ), "Check processing helper functions: process_expression !"
    assert (
        "gene_ontology" not in processed_data.columns
    ), "Check processing helper functions: process_expression !"

    # Check if the processed expression column has the correct values
    expected_values = [0.5, 0.8, 0.2]
    assert (
        processed_data["expression"].tolist() == expected_values
    ), "Check processing helper functions: process_expression !"


@pytest.mark.parametrize(
    ("data", "expected_columns"),
    [
        (
            pd.DataFrame(
                {
                    "A": ["[1, 2]", "[3, 4]", "[5, 6]"],
                    "B": ["7", "8", "9"],
                    "C": ["[10]", "[11]", "[12, 13]"],
                }
            ),
            {"A", "C"},
        ),
        (
            pd.DataFrame(
                {"A": ["[1, 2]", "[3, 4]", "[5, 6]"], "B": ["7", "8", "9"], "C": ["10", "11", "12"]}
            ),
            {"A"},
        ),
        (pd.DataFrame({"A": ["1", "2", "3"], "B": ["4", "5", "6"], "C": ["7", "8", "9"]}), set()),
    ],
)
def test_get_list_type(data: pd.DataFrame, expected_columns: Set) -> None:
    """This function is used to test the behavior of get_list_type function."""
    features_set_format: Set[str] = set()
    get_list_type(data, features_set_format)
    assert (
        features_set_format == expected_columns
    ), "Check processing helper functions: get_list_type !"


def test_preprocess_train(preprocess_train_df: pd.DataFrame) -> None:
    """This function is used to test the behavior of preprocess_train function."""
    expected_output = pd.DataFrame(
        {"A": [0, 1, 0, 1, 0, 5], "diff_class_prot": [True, 2, 3, 4, 5, False]}
    )

    processed_data = preprocess_train(preprocess_train_df)

    pd.testing.assert_frame_equal(
        processed_data, expected_output, obj="Check processing helper functions: preprocess_train !"
    )


@pytest.mark.parametrize(
    ("data", "col_name", "expected_type"),
    [
        (pd.DataFrame({"col": [1, 2, 3]}), "col", int),
        (pd.DataFrame({"col": [1.1, 2.2, 3.3]}), "col", float),
        (pd.DataFrame({"col": [True, False, True]}), "col", bool),
        (pd.DataFrame({"col": ["A", "B", "C"]}), "col", object),
    ],
)
def test_get_data_type(data: pd.DataFrame, col_name: str, expected_type: pd.Series.dtype) -> None:
    """This function is used to test the behavior of get_data_type function."""
    assert (
        get_data_type(data, col_name) == expected_type
    ), "Check processing helper functions: get_data_type !"


def test_get_features_type(
    mock_data_dict: Dict[str, pd.DataFrame], mock_features: List[str]
) -> None:
    """This function is used to test the behavior of get_features_type function."""
    expected_result = pd.DataFrame(
        {
            "Name": mock_features,
            "train": [int, float, object, bool],
            "test": [int, float, object, bool],
            "same_type": [True, True, True, True],
        }
    )

    result = get_features_type(mock_data_dict, mock_features)

    assert result.equals(expected_result), "Check processing helper functions: get_features_type !"


@pytest.mark.parametrize(
    ("type_to_get", "expected_result"),
    [
        (int, ["col1", "col4"]),
        (bool, ["col3", "col5"]),
        (float, ["col2", "col6"]),
        (object, []),
        (tuple, []),
    ],
)
def test_union_type_from_features_type(type_to_get: type, expected_result: List[str]) -> None:
    """This function is used to test the behavior of union_type_from_features_type function."""
    features_type = pd.DataFrame(
        {
            "Name": ["col1", "col2", "col3", "col4", "col5", "col6"],
            "train": [int, float, bool, int, bool, float],
            "test": [int, bool, bool, int, int, bool],
            "validation": [int, float, float, int, int, float],
        }
    )

    result = union_type_from_features_type(features_type, type_to_get)

    assert (
        result == expected_result
    ), "Check processing helper functions: union_type_from_features_type !"


def test_data_processor_single_data(mock_data: pd.DataFrame) -> None:
    """This function is used to test the behavior of data_processor_single_data function."""
    mock_configuration = MagicMock(
        filter_rows_value="filter_value",
        proxy_scores={"score1": "renamed_score1", "score2": "renamed_score2"},
        expression_raw_name="raw_expression",
        expression_name="processed_expression",
        include_features=["feature1", "feature2"],
        keep_include_features=False,
        label="target",
        nan_ratio=0.3,
        expression_filter="exp",
        base_columns=["id", "feature1"],
    )
    expected_data = pd.DataFrame(
        {
            "renamed_score1": [0.5, 0.3],
            "renamed_score2": [0.2, 0.6],
            "feature3": ["X", "Z"],
            "target": [0, 1],
            "second_id": [0, 1],
            "processed_expression": ["A", "C"],
        },
        index=[0, 1],
    )
    expected_base_data = pd.DataFrame(
        {"id": [1, 3], "feature1": [10, 30], "second_id": [0, 1]}, index=[0, 2]
    )

    processed_data = data_processor_single_data(mock_data, mock_configuration)

    pd.testing.assert_frame_equal(
        processed_data["proc"],
        expected_data,
        obj="Check processing helper functions: data_processor_single_data !",
    )
    pd.testing.assert_frame_equal(
        processed_data["base"],
        expected_base_data,
        obj="Check processing helper functions: data_processor_single_data !",
    )


def test_get_features_from_features_configuration() -> None:
    """This function is used to test the behavior of get_features_from_features_configuration function."""
    features_configuration = {
        "ids": ["id1", "id2"],
        "float": ["float1", "float2"],
        "int": ["int1", "int2"],
        "categorical": ["categorical1", "categorical2"],
        "bool": ["bool1", "bool2"],
        "label": "label",
    }

    expected_features = [
        "id1",
        "id2",
        "float1",
        "float2",
        "int1",
        "int2",
        "categorical1",
        "categorical2",
        "bool1",
        "bool2",
        "label",
    ]

    features = get_features_from_features_configuration(features_configuration)

    assert (
        features == expected_features
    ), "Check processing helper functions: get_features_from_features_configuration !"


def test_cross_validation() -> None:
    """This function is used to test the behavior of cross_validation function."""
    train_data = pd.DataFrame({"proc_sec_id": [1, 2, 3], "second_id": [0, 1, 2]})

    configuration = MagicMock(
        split=True,
        split_nfold=3,
        split_seed=1234,
        split_val_size=0.2,
        split_kfold_column_name="kfold",
        split_train_val_name="train_val",
        output_dir=Path("tests/fixtures"),
    )

    train_data, cv_columns = cross_validation(train_data=train_data, configuration=configuration)

    assert cv_columns == ["kfold", "train_val"]
    assert train_data["kfold"].dtype == np.dtype("int64")
    assert train_data["train_val"].dtype == np.dtype("int64")

    assert train_data.kfold.unique().tolist() == [0, 1, 2]
    assert train_data.train_val.unique().tolist() == [1, 0]
    assert "splits.csv" in os.listdir(
        configuration.output_dir
    ), "Check processing helper functions: cross_validation !"


def test_find_duplicated_columns_by_values() -> None:
    """This function is used to test the behavior of find_duplicated_columns_by_values function."""
    features = ["col1", "col2", "col3", "col4"]
    include_keywords = ["col1", "col4"]

    dataframe = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [0, 2, 3, 4, 5],
            "col3": [1, 2, 3, 4, 5],
            "col4": [1, 2, 3, 4, 5],
        }
    )
    data = {"train": {"proc": dataframe}}

    duplicated_columns = find_duplicated_columns_by_values(data, features, include_keywords)

    assert set(duplicated_columns) == {
        "col3",
        "col1",
    }, "Check processing helper functions: find_duplicated_columns_by_values !"


def test_get_duplicate_columns() -> None:
    """This function is used to test the behavior of get_duplicate_columns function."""
    features = ["col1", "col2", "col3", "col4", "col5"]
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1, 2, 3, 4, 5],
            "col3": [1, 2, 3, 4, 5],
            "col4": [1, 2, 3, 4, 5],
            "col5": [1, 2, 3, 4, 5],
        }
    )

    duplicated_columns = get_duplicate_columns(data, features)

    assert duplicated_columns == {
        "col1": ["col2", "col3", "col4", "col5"]
    }, "Check processing helper functions: get_duplicate_columns !"


@pytest.mark.parametrize(
    (
        "duplicated_columns",
        "main_columns",
        "expected_duplicated_columns",
        "expected_duplicated_to_remove",
    ),
    [
        ({}, [], {}, []),
        ({"col1": ["col1", "col2"]}, [], {"col2": ["col1", "col1"]}, ["col1", "col1"]),
        (
            {
                "col1": ["col1", "col2"],
                "col2": ["col2", "col3"],
                "col3": ["col3", "col4"],
            },
            ["col1", "col2", "col3"],
            {"col2": ["col1", "col1"], "col3": ["col2", "col2"], "col4": ["col3", "col3"]},
            [],
        ),
    ],
)
def test_sort_duplicated_columns(
    duplicated_columns: Dict[str, List[str]],
    main_columns: List[str],
    expected_duplicated_columns: Dict[str, List[str]],
    expected_duplicated_to_remove: List[str],
) -> None:
    """This function is used to test the behavior of sort_duplicated_columns function."""
    result_duplicated_columns, result_duplicated_to_remove = sort_duplicated_columns(
        duplicated_columns, main_columns
    )
    assert (
        result_duplicated_columns == expected_duplicated_columns
    ), "Check processing helper functions: sort_duplicated_columns !"
    assert (
        result_duplicated_to_remove == expected_duplicated_to_remove
    ), "Check processing helper functions: sort_duplicated_columns !"


@pytest.mark.parametrize(
    ("data", "main_columns", "expected_columns"),
    [
        (pd.DataFrame(), [], []),
        (pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}), [], ["col1", "col2"]),
        (
            pd.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": [4, 5, 6],
                    "col3": [1, 2, 3],
                    "col4": [1, 1, 2],
                    "col5": [1, 2, 3],
                }
            ),
            ["col2", "col5"],
            ["col2", "col4", "col5"],
        ),
    ],
)
def test_remove_duplicate_columns(
    data: pd.DataFrame, main_columns: List[str], expected_columns: List[str]
) -> None:
    """This function is used to test the behavior of remove_duplicate_columns function."""
    result = remove_duplicate_columns(data, main_columns)
    assert (
        list(result.columns) == expected_columns
    ), "Check processing helper functions: remove_duplicate_columns !"


def test_track_features() -> None:
    """This function is used to test the behavior of track_features function."""
    features = ["col1", "col2", "col3"]
    output_dir = CONFIGURATION_DIRECTORY

    track_features(features, output_dir)

    feature_tracker_path = CONFIGURATION_DIRECTORY / "features_tracker.txt"
    with open(feature_tracker_path) as f:
        lines = f.readlines()
    assert feature_tracker_path.exists(), "Check processing helper functions: track_features !"

    assert lines == [
        "c o l 1\n",
        "c o l 2\n",
        "c o l 3\n",
    ], "Check processing helper functions: track_features !"


@pytest.mark.parametrize(
    (
        "type_name",
        "features_configuration",
        "features",
        "include_features",
        "base_features",
        "take",
        "take_from_include_features",
        "final_common_features",
        "expected_config",
        "expected_final_common",
    ),
    [
        (
            "test_type_1",
            {},
            ["feature1", "feature2", "feature3"],
            ["feature1", "feature2", "feature4"],
            ["feature1"],
            True,
            False,
            [],
            {"test_type_1": ["feature1", "feature1", "feature2"]},
            ["feature1", "feature1", "feature2"],
        ),
        (
            "test_type_2",
            {},
            ["feature1", "feature2", "feature3"],
            ["feature1", "feature2", "feature4"],
            ["feature1"],
            False,
            False,
            [],
            {},
            [],
        ),
    ],
)
def test_add_features(
    type_name: str,
    features_configuration: Dict[str, Any],
    features: List[str],
    include_features: List[str],
    base_features: List[str],
    take: bool,
    take_from_include_features: bool,
    final_common_features: List[str],
    expected_config: Dict[str, Any],
    expected_final_common: List[str],
) -> None:
    """This function is used to test the behavior of add_features function."""
    add_features(
        type_name,
        features_configuration,
        features,
        include_features,
        base_features,
        take,
        take_from_include_features,
        final_common_features,
    )

    assert (
        features_configuration == expected_config
    ), "Check processing helper functions: add_features !"
    assert (
        final_common_features == expected_final_common
    ), "Check processing helper functions: add_features !"


def test_legend_replace_renamed_columns() -> None:
    """This function is used to test the behavior of legend_replace_renamed_columns function."""
    legend_features = ["feature1", "feature2", "feature3"]

    configuration = MagicMock()
    configuration.proxy_scores = {"feature1": "renamed_feature1", "feature2": "renamed_feature2"}
    configuration.expression_raw_name = "expression_raw"
    configuration.expression_name = "expression"

    result = legend_replace_renamed_columns(legend_features, configuration)

    expected_result = ["feature3", "renamed_feature1", "renamed_feature2"]

    assert (
        result == expected_result
    ), "Check processing helper functions: legend_replace_renamed_columns !"


def test_data_processor_new_data(mock_configuration: MagicMock) -> None:
    """This function is used to test the behavior of data_processor_new_data function."""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "score1": [0.1, 0.2, 0.3],
            "score2": [0.4, 0.5, 0.6],
            "second_id": [0, 1, 2],
            "expression_column": ["A", "B", "C"],
            "target": [0, "1", 0],
        }
    )

    result = data_processor_new_data(data, mock_configuration)

    expected_result = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "renamed_score1": [0.1, 0.2, 0.3],
            "renamed_score2": [0.4, 0.5, 0.6],
            "second_id": [0, 1, 2],
            "target": [0, 1, 0],
            "expression_column": ["A", "B", "C"],
        }
    )
    pd.testing.assert_frame_equal(
        result, expected_result, obj="Check processing helper functions: data_processor_new_data !"
    )


@pytest.mark.parametrize(
    ("data", "features", "ignore_missing_features", "expected_exit_code"),
    [
        (pd.DataFrame(), [], False, Exception),
        (
            pd.DataFrame({"id1": [1, 2], "id2": [3, 4], "float1": [5, 6], "float2": [7, 8]}),
            [],
            False,
            Exception,
        ),
        (
            pd.DataFrame(),
            [
                "id1",
                "id2",
                "float1",
                "float2",
                "int1",
                "int2",
                "categorical1",
                "categorical2",
                "bool1",
                "bool2",
            ],
            False,
            0,
        ),
        (pd.DataFrame(), ["id1", "id2"], False, 1),
        (
            pd.DataFrame({"id1": [1, 2], "id2": [3, 4], "float1": [5, 6], "float2": [7, 8]}),
            [
                "id1",
                "id2",
                "float1",
                "float2",
                "int1",
                "int2",
                "categorical1",
                "categorical2",
                "bool1",
                "bool2",
            ],
            True,
            0,
        ),
    ],
)
def test_report_missing_columns(
    data: pd.DataFrame,
    features: List[str],
    ignore_missing_features: bool,
    expected_exit_code: Union[str, int],
) -> None:
    """This function is used to test the behavior of report_missing_columns function."""
    # pylint: disable=broad-except
    try:
        report_missing_columns(data, features, ignore_missing_features)
    except SystemExit as e:
        # flake8: noqa PT017
        assert (
            e.code == expected_exit_code
        ), "Check processing helper functions: report_missing_columns !"

    except Exception as e:
        assert (
            e.__str__() == "Features from features configuration are missing"
        ), "Check processing helper functions: report_missing_columns !"
