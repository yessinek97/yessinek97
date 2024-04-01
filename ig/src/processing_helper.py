"""This module provides some helper functions to the data processing and features selection."""
import collections
import re
from collections import defaultdict
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from ig import DATA_DIRECTORY, PROC_SEC_ID
from ig.src.logger import get_logger
from ig.src.utils import load_yml, read_data, save_yml

log: Logger = get_logger("Processing_help")


def load_dataset(data_path: str, **kwargs: Any) -> pd.DataFrame:
    """This function is used to read the input dataset and lower column names.

    Args:
        data_path: Path of the input dataset.

    Returns:
        data: The input dataframe.
    """
    data = read_data(data_path, **kwargs)
    data.columns = data.columns.str.lower()
    return data


class LoadProcessingConfiguration:
    """Load and parse the configuration file."""

    def __init__(
        self, config_path: Optional[Path] = None, output_dir: Optional[Path] = None
    ) -> None:
        """Init method."""
        if config_path:
            self.configuration = load_yml(config_path)
            self.output_dir = DATA_DIRECTORY / f"proc_data_{self.version}_{self.data_type}"
        if output_dir:
            self.output_dir = output_dir
            self.configuration = load_yml(self.output_dir / "configuration.yml")
        self.legend_path = self.legend.get("path", None)
        self.legend_filter_column = self.legend.get("filter_column", None)
        self.legend_value = self.legend.get("value", None)
        self.legend_feat_col_name = self.legend.get("feat_col_name", None)
        self.label = self.processing["label"].lower()

        self.id = str(self.processing["id"]).lower()
        self.ids = [col.lower() for col in self.processing.get("ids", [])]

        self.proxy_m_pep = self.proxy_model_columns.get("proxy_m_peptide", None)
        if self.proxy_m_pep:
            self.proxy_m_pep = self.proxy_m_pep.lower()
        self.proxy_wt_pep = self.proxy_model_columns.get("proxy_wt_peptide", None)
        if self.proxy_wt_pep:
            self.proxy_wt_pep = self.proxy_wt_pep.lower()
        self.proxy_allele = self.proxy_model_columns.get("proxy_allele", None)
        if self.proxy_allele:
            self.proxy_allele = self.proxy_allele.lower()

        self.proxy_scores = self.proxy_model_columns.get("scores", {})
        self.proxy_scores = {
            k.lower(): v for k, v in zip(self.proxy_scores.keys(), self.proxy_scores.values())
        }
        self.expression_filter = [col.lower() for col in self.expression.get("filter", [])]
        self.expression_raw_name = self.expression.get("raw_name", None)
        if self.expression_raw_name:
            self.expression_raw_name = self.expression_raw_name.lower()

        self.expression_name = self.expression.get("name")

        self.exclude_features = [col.lower() for col in self.processing.get("exclude_features", [])]
        self.include_features = [col.lower() for col in self.processing.get("include_features", [])]
        self.keep_include_features = self.processing.get("keep_include_features", False)

        self.filter_rows_column = self.filter_rows.get("filter_column", None)
        if self.filter_rows_column:
            self.filter_rows_column = self.filter_rows_column.lower()

        self.filter_rows_value = self.filter_rows.get("value", None)

        self.nan_ratio = self.processing.get("nan_ratio", 0.6)

        self.features_file_name = self.features.get("file_name", "features.yml")

        self.features_float = self.features.get("float", False)
        self.features_include_float = self.features.get("include_features_float", False)

        self.features_int = self.features.get("int", False)
        self.features_include_int = self.features.get("include_features_int", False)

        self.features_bool = self.features.get("bool", False)
        self.features_include_bool = self.features.get("include_features_bool", False)

        self.features_categorical = self.features.get("categorical", False)
        self.features_include_categorical = self.features.get("include_features_categorical", False)

        self.split_kfold = self.split.get("kfold", True)
        self.split_kfold_column_name = self.split.get("kfold_column_name", "fold")
        self.split_nfold = self.split.get("nfold", 5)
        self.split_train_val = self.split.get("train_val", True)
        self.split_train_val_name = self.split.get("train_val_name", "validation")
        self.split_val_size = self.split.get("val_size", 0.1)
        self.split_source = self.split.get("source_split", False)
        self.split_source_column = self.split.get("source_column", "author/source")
        self.split_seed = self.split.get("seed", 2023)
        self.base_columns = self.get_base_column()

    @property
    def version(self) -> str:
        """Return version parameter  from configuration."""
        return self.configuration.get("data_version", "")

    @property
    def data_type(self) -> str:
        """Return type parameter from configuration."""
        return self.configuration.get("data_type", "")

    @property
    def push_gcp(self) -> bool:
        """Return push_gcp parameter from configuration."""
        return self.configuration.get("push_gcp", False)

    @property
    def processing(self) -> Dict[str, Any]:
        """Return processing section from configuration."""
        return self.configuration["processing"]

    @property
    def legend(self) -> Dict[str, Any]:
        """Return legend section from configuration."""
        return self.processing.get("legend", {})

    @property
    def proxy_model_columns(self) -> Dict[str, Any]:
        """Return proxy_model_columns section from configuration."""
        return self.processing.get("proxy_model_columns", {})

    @property
    def expression(self) -> Dict[str, Any]:
        """Return expression section from configuration."""
        return self.processing.get("expression", {})

    @property
    def features(self) -> Dict[str, Any]:
        """Return features section from configuration."""
        return self.configuration.get("features", {})

    @property
    def split(self) -> Dict[str, Any]:
        """Return split section from configuration."""
        return self.configuration.get("split", {})

    @property
    def filter_rows(self) -> Dict[str, Any]:
        """Return filter_rows section from configuration."""
        return self.processing.get("filter_rows", {})

    def save_configuration(self) -> None:
        """Save features configuration."""
        save_yml(self.configuration, self.output_dir / "configuration.yml")

    def get_base_column(self) -> List[str]:
        """Generate the base columns features."""
        base_columns = self.ids
        if self.proxy_m_pep:
            base_columns.append(self.proxy_m_pep)
        if self.proxy_wt_pep:
            base_columns.append(self.proxy_wt_pep)
        if self.proxy_allele:
            base_columns.append(self.proxy_allele)
        base_columns.append(self.label)
        if self.id not in base_columns:
            base_columns = [self.id] + base_columns
        return base_columns


def remove_rows(data: pd.DataFrame, configuration: LoadProcessingConfiguration) -> pd.DataFrame:
    """Remove rows from data based on the filter."""
    if (configuration.filter_rows_column is not None) and (
        configuration.filter_rows_value is not None
    ):
        if configuration.filter_rows_column in data.columns:
            log.info(
                "-- Remove rows using  %s column  with the value %s",
                configuration.filter_rows_column,
                configuration.filter_rows_value,
            )
            log.info("--- data shape before  removing rows %s", data.shape)
            data = data[data[configuration.filter_rows_column] != configuration.filter_rows_value]
            log.info("--- data shape after   removing rows %s", data.shape)
        else:
            log.info("[%s] column is not defined in the data", configuration.filter_rows_column)
    return data


def replace_nan_strings(data: pd.DataFrame) -> pd.DataFrame:
    """Replace nan strings with numpy nan.

    Args:
        data: dataframe to replace its nan strings.

    Returns:
        data: The new dataframe.
    """
    log.info("--- initial %s null values", data.isna().sum().sum())
    data = data.replace("\n", "", regex=True)
    data = data.replace("\t", "", regex=True)
    nan_strings = ["NA", "NaN", "NAN", "NE", "n.d.", "nan", "na", "?", ""]
    for col in data.columns:
        for nanstr in nan_strings:
            data[col] = data[col].replace(nanstr, np.nan)
    log.info("--- final %s null values", data.isna().sum().sum())
    return data


def clean_target(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Clean the dataframe's target.

    Args:
    data: dataframe to clean its target.
    target: the target's column name.

    Returns:
    data: dataframe with cleaned target.
    """
    log.info("--- data shape before process label %s", data.shape)
    data = data[data[target].isin(["0", "1", 0, 1])]
    data[target] = data[target].astype(int)
    log.info("--- data shape after  process label %s", data.shape)
    return data


def rename_scores(data: pd.DataFrame, scores: Dict[str, str]) -> pd.DataFrame:
    """Rename scores columns."""
    renamed_columns = [col for col in data.columns if col in scores]
    data = data.rename(columns=scores)
    if len(renamed_columns):
        log.info("--- %s columns are renamed", len(renamed_columns))
    return data


def get_columns_by_keywords(columns_list: List[str], keywords: Union[List[str], str]) -> List[str]:
    """Filter columns by keywords."""
    cols = []

    columns_list = [col.lower() for col in columns_list]
    if len(keywords):

        if isinstance(keywords, list):
            words = re.compile("|".join(keywords))
        else:
            words = re.compile(keywords)

        for col in columns_list:
            match = words.search(col)
            if match:
                cols.append(col)

    return cols


def remove_columns(
    data: pd.DataFrame,
    columns: List[str],
    name: str = "",
    keep_features: bool = False,
    include_features: Optional[List[str]] = None,
    process_name: Optional[str] = "",
) -> pd.DataFrame:
    """Remove columns from data."""
    log.info("--- %sdata shape before removing columns %s", name, data.shape)
    removed_include_features = []
    if include_features:
        removed_include_features = [col for col in include_features if col in columns]
        if keep_features:
            columns = [col for col in columns if col not in include_features]
            log.info(
                "--- %s features are kept from the include features during %s",
                len(removed_include_features),
                process_name,
            )
            for i in range(0, len(removed_include_features), 3):
                log.info("---- %s", " | ".join(removed_include_features[i : i + 3]))
        else:
            log.info(
                "--- %s is removed from the include features during  %s",
                len(removed_include_features),
                process_name,
            )
    log.info("--- %s features removed from data", len(columns))
    data = data.drop([col for col in columns if col in data.columns], axis=1)
    log.info("--- %s data shape after  removing columns %s", name, data.shape)
    return data


def remove_duplicated_columns_by_name(data: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with the same name."""
    unique_columns = data.columns.value_counts() > 1
    duplicate_to_drop = list(unique_columns[unique_columns].index)
    log.info("--- %s duplicated columns", len(duplicate_to_drop))
    for i in range(0, len(duplicate_to_drop), 3):
        log.info("---- %s", " | ".join(duplicate_to_drop[i : i + 3]))
    data = remove_columns(data, duplicate_to_drop)
    return data


def remove_columns_with_unique_value(data: pd.DataFrame) -> pd.DataFrame:
    """Get columns with one unique value."""
    df = data_description(data, data.columns)
    columns_with_unique_value = set(df[df["Nunique"] == 1].Name)
    log.info(
        "--- data has %s columns. %s of which have one unique value.",
        len(data.columns),
        len(columns_with_unique_value),
    )
    data = remove_columns(data, list(columns_with_unique_value))
    return data


def remove_nan_columns(
    data: pd.DataFrame, nan_ratio: float, keep_features: bool, include_features: List[str]
) -> pd.DataFrame:
    """Get columns with full nans."""
    data_des = data_description(data, data.columns)
    data_full_nan_columns = data_des[data_des["Nan Ratio"] >= nan_ratio].Name.tolist()
    log.info(
        "--- full nan columns: %s | %s",
        len(data_full_nan_columns),
        len(data_full_nan_columns) / data.shape[1],
    )
    data = remove_columns(
        data,
        data_full_nan_columns,
        keep_features=keep_features,
        include_features=include_features,
        process_name="remove_nan_columns",
    )
    return data


def remove_list_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Get and remove columns that have list format."""
    features_list_format = []
    for col in data.columns:
        ratio = data[col].apply(is_list).mean()
        if ratio:
            features_list_format.append(col)
    log.info("--- features list format to drop: %s", len(features_list_format))
    data = remove_columns(data, features_list_format)
    return data


def process_expression(
    data: pd.DataFrame,
    configuration: LoadProcessingConfiguration,
    include_features: List[str],
) -> pd.DataFrame:
    """Pop expression columns from data and merge the real expression column to the given data."""
    expression_columns = get_columns_by_keywords(
        data.columns.to_list(), configuration.expression_filter
    )
    log.info("--- expression columns : %s", len(expression_columns))

    expression_data = data[[PROC_SEC_ID] + expression_columns].copy()
    data = remove_columns(
        data,
        expression_columns,
        keep_features=True,
        include_features=include_features,
        process_name="process expression",
    )
    expression_data.rename(
        columns={configuration.expression_raw_name: configuration.expression_name}, inplace=True
    )

    data = data.merge(
        expression_data[[PROC_SEC_ID, configuration.expression_name]], how="left", on=[PROC_SEC_ID]
    )

    return data


def data_processor_single_data(
    data: pd.DataFrame, configuration: LoadProcessingConfiguration
) -> Dict[str, pd.DataFrame]:
    """Helper function to apply minor cleaning steps."""
    log.info("-- Replace nan strings")
    data = replace_nan_strings(data)
    log.info("-- Process label")
    data = clean_target(data, configuration.label.lower())
    data = remove_rows(data, configuration)
    if configuration.proxy_scores:
        log.info("-- Rename score")
        data = rename_scores(data, configuration.proxy_scores)
    include_features = get_columns_by_keywords(
        data.columns.to_list(), configuration.include_features
    )
    data[PROC_SEC_ID] = range(len(data))
    base_data = data[configuration.base_columns + [PROC_SEC_ID]]
    log.info("*" * 50)
    log.info("-- Remove base features")
    data = remove_columns(data, configuration.base_columns)
    log.info("*" * 50)
    log.info("-- Remove duplicated columns by name")
    data = remove_duplicated_columns_by_name(data)
    log.info("*" * 50)
    log.info("-- Remove columns with unique value")
    data = remove_columns_with_unique_value(data)
    log.info("*" * 50)
    log.info("-- Remove features with more than %s nan ratio", configuration.nan_ratio)
    data = remove_nan_columns(
        data,
        configuration.nan_ratio,
        include_features=include_features,
        keep_features=configuration.keep_include_features,
    )
    log.info("*" * 50)
    log.info("-- Remove features with list type ")
    data = remove_list_columns(data)
    log.info("*" * 50)
    if configuration.expression:
        log.info("-- Process expression column")
        data = process_expression(data, configuration, include_features)
    else:
        log.info("-- Process expression column no configuration was provided.")

    return {"proc": data, "base": base_data}


def get_list_type(data: pd.DataFrame, features_set_format: Set[str]) -> None:
    """Get columns that have list format."""
    for col in data.columns:
        ratio = data[col].apply(is_list).mean()
        if ratio:
            features_set_format.add(col)


def data_description(data: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """Produce some numerical descriptions to the data."""
    output = pd.DataFrame()
    output["Name"] = names
    output["Type"] = output.Name.map(data.dtypes)
    output["Min"] = output.Name.map(data.min(axis=0, numeric_only=True))
    output["Max"] = output.Name.map(data.max(axis=0, numeric_only=True))
    output["Mean"] = output.Name.map(data.mean(axis=0, numeric_only=True))
    output["Nan count"] = output.Name.map(data.isna().sum())
    output["Nan Ratio"] = output.Name.map(data.isna().mean())
    output["Nunique"] = output.Name.map(data.nunique())
    output["Unique"] = [data[col].unique() for col in names]
    return output


def is_list(x: str) -> int:
    """Check if the string x is actually a list."""
    try:
        if x.startswith("[") and x.endswith("]"):
            return 1
        return 0
    except AttributeError:
        return 0


def is_integer(x: Any) -> bool:
    """Check whether or not x is an integer."""
    # pylint: disable=unidiomatic-typecheck
    return type(x) is int


def is_bool(x: Any) -> bool:
    """Check whether or not x is a boolean."""
    return isinstance(x, bool)


def get_features_type(
    data: Dict[str, pd.DataFrame],
    features: List[str],
) -> pd.DataFrame:
    """Get features type from all datasets."""
    features_type = pd.DataFrame()
    features_type["Name"] = features
    data["train"] = preprocess_train(data["train"])
    for name in data.keys():
        features_type[name] = [get_data_type(data[name], col) for col in features]

    features_type["same_type"] = (
        features_type[list(data.keys())]
        .eq(features_type[list(data.keys())[0]].iloc[:], axis=0)
        .all(1)
    )
    log.info("--- ratio of features with the same type  = %s", features_type.same_type.mean())
    features_type = features_type[features_type["same_type"] == 1]
    return features_type


def get_data_type(data: pd.DataFrame, col_name: str) -> Type:
    """Return the data type of column."""
    col = data[col_name].dropna()
    if col.apply(is_integer).all():
        return int

    if col.apply(is_bool).all():
        return bool
    return col.dtype


def preprocess_train(public: pd.DataFrame) -> pd.DataFrame:
    """Apply minor preprocessing."""
    log.info("--- Function: preprocess_public")
    public.diff_class_prot = public.diff_class_prot.replace({0: False, 1: True})
    return public


def union_type_from_features_type(features_type: pd.DataFrame, type_to_get: Type) -> List[str]:
    """Return list of features from features_type that have the same type as to type_to_get."""
    optima_type = features_type[features_type.train == type_to_get].Name.tolist()
    columns = optima_type
    columns.sort()
    return columns


def data_processor_new_data(
    data: pd.DataFrame, configuration: LoadProcessingConfiguration
) -> pd.DataFrame:
    """Helper function to apply minor cleaning steps."""
    log.info("-- Replace nan strings")
    data = replace_nan_strings(data)
    log.info("-- Process label")
    data = clean_target(data, configuration.label)
    data[PROC_SEC_ID] = range(len(data))
    if configuration.proxy_scores:
        log.info("-- Rename score")
        data = rename_scores(data, configuration.proxy_scores)
    include_features = get_columns_by_keywords(
        data.columns.to_list(), configuration.include_features
    )
    if configuration.expression:
        log.info("-- Process expression column")
        data = process_expression(data, configuration, include_features)
    else:
        log.info("-- Process expression column no configuration was provided.")

    return data


def get_features_from_features_configuration(features_configuration: Dict[str, Any]) -> List[str]:
    """Extract the features and return them in list format."""
    features = features_configuration["ids"]
    if "float" in features_configuration.keys():
        features.extend(features_configuration["float"])
    if "int" in features_configuration.keys():
        features.extend(features_configuration["int"])
    if "categorical" in features_configuration.keys():
        features.extend(features_configuration["categorical"])
    if "bool" in features_configuration.keys():
        features.extend(features_configuration["bool"])

    features.append(features_configuration["label"])
    return features


def report_missing_columns(
    data: pd.DataFrame, features: List[str], ignore_missing_features: bool
) -> None:
    """Report the missing features."""
    missing_columns = [col for col in features if col not in data.columns]
    if missing_columns:
        log.info("*" * 50)
        log.info("- %s features are missing from features configuration", len(missing_columns))
        for i in range(0, len(missing_columns), 3):
            log.info("-- %s", " | ".join(missing_columns[i : i + 3]))
        log.info("*" * 50)
        if not ignore_missing_features:
            raise Exception("Features from features configuration are missing")


def cross_validation(
    train_data: pd.DataFrame, configuration: LoadProcessingConfiguration
) -> Tuple[pd.DataFrame, List[str]]:
    """Add corss validation columns, Kfold and train val split."""
    cv_columns = []
    if configuration.split:
        log.info("- Add cross validation columns to Train data")
        splits = train_data[[PROC_SEC_ID]]
        splits.reset_index(inplace=True, drop=True)
        if configuration.split_kfold:
            cv_columns.append(configuration.split_kfold_column_name)
            splits[configuration.split_kfold_column_name] = 0
            for i, (_, val_index) in enumerate(
                KFold(
                    configuration.split_nfold, random_state=configuration.split_seed, shuffle=True
                ).split(X=splits[PROC_SEC_ID])
            ):
                splits.loc[val_index, configuration.split_kfold_column_name] = int(i)

        if configuration.split_train_val:
            cv_columns.append(configuration.split_train_val_name)

            _, val_index = train_test_split(
                splits[PROC_SEC_ID],
                random_state=configuration.split_seed,
                test_size=configuration.split_val_size,
            )
            splits[configuration.split_train_val_name] = 0
            splits.loc[splits[PROC_SEC_ID].isin(val_index), configuration.split_train_val_name] = 1

        splits.to_csv(configuration.output_dir / "splits.csv", index=False)
        train_data = train_data.merge(splits, on=PROC_SEC_ID, how="left")

    return train_data, cv_columns


def find_duplicated_columns_by_values(  # noqa
    data: Dict[str, Dict[str, pd.DataFrame]],
    features: List[str],
    include_keywords: List[str],
) -> List[str]:
    """Find duplicated columns by values."""
    log.info("- Fixing duplicated columns by values")
    dict_duplicated_columns_set = collections.defaultdict(set)
    dict_duplicated_columns: Dict[str, List[str]] = {}
    for name in data:
        log.info("-- %s :", name)
        data[name]["duplicated_columns"] = get_duplicate_columns(data[name]["proc"], features)
        log.info("--- %s duplicated columns", len(data[name]["duplicated_columns"]))
        for k, v in data[name]["duplicated_columns"].items():  # d.items() in Python 3+
            dict_duplicated_columns_set[k].update(v)
    log.info("-- Find the intersection between dataset")
    for key in dict_duplicated_columns_set:
        dict_duplicated_columns[key] = list(dict_duplicated_columns_set[key])

    keys = list(dict_duplicated_columns.keys())
    second_keys = deepcopy(keys)
    for key in keys:
        if key in second_keys:
            second_keys.remove(key)
            keys_to_drop = []
            for second_key in second_keys:

                if key in dict_duplicated_columns[second_key]:
                    dict_duplicated_columns[key].extend(
                        [col for col in dict_duplicated_columns[second_key] if col != key]
                    )
                    dict_duplicated_columns[key].append(second_key)
                    dict_duplicated_columns.pop(second_key)
                    keys_to_drop.append(second_key)
                    dict_duplicated_columns[key] = list(set(dict_duplicated_columns[key]))
            second_keys = [e for e in second_keys if e not in keys_to_drop]

    include_features = get_columns_by_keywords(features, include_keywords)
    duplicated_from_include = {}
    keys = list(dict_duplicated_columns.keys())
    for key in keys:
        intersection = list(set(dict_duplicated_columns[key]) & set(include_features))
        if intersection:
            duplicated = dict_duplicated_columns[key]
            duplicated.remove(intersection[0])
            duplicated.append(key)
            duplicated_from_include[intersection[0]] = duplicated
            dict_duplicated_columns.pop(key)
    dict_duplicated_columns.update(duplicated_from_include)

    duplicated_columns = []
    for key in dict_duplicated_columns:
        duplicated_columns.extend(dict_duplicated_columns[key])
    duplicated_columns = list(set(duplicated_columns))
    log.info("-- %s are duplicated", len(duplicated_columns))
    return duplicated_columns


def remove_duplicate_columns(
    data: pd.DataFrame, main_columns: List[str], name: str = ""
) -> pd.DataFrame:
    """Remove duplicate columns from data."""
    duplicated_columns = get_duplicate_columns(data, data.columns)
    sorted_duplicated_columns, duplicated_to_remove = sort_duplicated_columns(
        duplicated_columns, main_columns
    )
    log.info(name)
    for key in sorted_duplicated_columns:
        sorted_msg = " ".join(sorted_duplicated_columns[key])
        log.info("%s : %s", key, sorted_msg)
    data = remove_columns(data, duplicated_to_remove, name)
    return data


def get_duplicate_columns(data: pd.DataFrame, features: List[str]) -> Dict[str, List[str]]:
    """Get duplicate columns from data."""
    all_duplicates_columns = []
    duplicate_column_names = defaultdict(list)
    columns = features
    for pos, column in enumerate(columns):
        if column not in all_duplicates_columns:
            column_value = data[column]
            for other_column in columns[pos + 1 :]:
                other_column_value = data[other_column]
                if column_value.equals(other_column_value):
                    duplicate_column_names[column].append(other_column)
                    all_duplicates_columns.append(other_column)
    return duplicate_column_names


def sort_duplicated_columns(
    columns: Dict[str, List[str]], main_columns: List[str]
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Sort duplicated columns."""
    duplicated_columns = {}
    duplicated_to_remove = []
    for key in columns:
        values = np.sort([key] + columns[key])
        duplicated_columns[values[-1]] = list(values[0:-1])
        duplicated_to_remove.extend(list(values[0:-1]))
    duplicated_to_remove = [col for col in duplicated_to_remove if col not in main_columns]
    return duplicated_columns, duplicated_to_remove


def track_features(features: List[str], output_dir: Path) -> None:
    """Track  features and save them into a file."""
    with open(output_dir / "features_tracker.txt", "w") as file:
        for row in features:
            s = " ".join(map(str, row))
            file.write(s + "\n")


def add_features(
    type_name: str,
    features_configuration: Dict[str, Any],
    features: List[str],
    include_features: List[str],
    base_features: List[str],
    take: bool,
    take_from_include_features: bool,
    final_common_features: List[str],
) -> None:
    """Add a specific features type to features configuration."""
    fs = [col for col in features if col in include_features]
    if take:
        features_configuration[type_name] = [col for col in features if col in base_features] + fs
        final_common_features.extend(features_configuration[type_name])
        log.info("-- Add %s features %s", type_name, len(features_configuration[type_name]))
    else:
        if take_from_include_features and (len(fs) > 0):
            features_configuration[type_name] = fs
            final_common_features.extend(features_configuration[type_name])
            log.info("-- Add %s features %s", type_name, len(features_configuration[type_name]))


def legend_replace_renamed_columns(
    legend_features: List[str], configuration: LoadProcessingConfiguration
) -> List[str]:
    """Replace renamed columns if they are include in the legend file."""
    for feature in configuration.proxy_scores:
        if feature in legend_features:
            legend_features.remove(feature)
            legend_features.append(configuration.proxy_scores[feature])
    if configuration.expression_raw_name is not None and configuration.expression_name is not None:
        if configuration.expression_raw_name in legend_features:
            legend_features.remove(configuration.expression_raw_name)
            legend_features.append(configuration.expression_name)
    return legend_features
