"""Holds functions for data preprocessing."""
from __future__ import annotations

import re
from logging import Logger
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split

from ig import DATA_DIRECTORY, DATAPROC_DIRECTORY, KFOLD_EXP_NAMES, SINGLE_MODEL_NAME
from ig.bucket.click.constants import GS_BUCKET_PREFIX
from ig.buckets import pull
from ig.src.logger import get_logger
from ig.src.utils import load_pkl, load_yml, read_data, remove_bucket_prefix, save_as_pkl, save_yml

log: Logger = get_logger("Processing")


class Dataset:
    """Class to handle data processing and cleaning."""

    # ----------------------------------------------
    def __init__(
        self,
        data_path: str,
        configuration: dict[str, Any],
        is_train: bool,
        experiment_path: Path,
        click_ctx: click.core.Context | Any,
        is_unlabeled: bool = False,
        force_gcp: bool = False,
        is_inference: bool = False,
        process_label: bool = True,
    ):
        """Initialize the Dataset class."""
        self.is_train = is_train
        self.experiment_path = experiment_path
        self.is_unlabeled = is_unlabeled
        self.configuration = configuration
        self.experiment_proc_data_folder = self.experiment_path / DATAPROC_DIRECTORY
        self.experiment_proc_data_folder.mkdir(exist_ok=True, parents=True)
        self.splits_columns: list[str] = []
        self.data_path = self.maybe_pull_data_from_gcp(data_path, click_ctx, force_gcp)
        self.features_configuration = self.load_features_configuration(click_ctx, force_gcp)
        self.dataset_name = f"{Path(data_path).parent.name}_data"
        self.all_features = self.float_features + self.int_features + self.categorical_features
        self.use_validation_strategy = self.processing_configuration.get(
            "validation_strategy", True
        )
        self.is_inference = is_inference
        self.process_label = process_label
        self.features: list[str]

    @property
    def processing_configuration(self) -> dict[str, Any]:
        """Return processing configuration."""
        return self.configuration["processing"]

    @property
    def label(self) -> str:
        """Return label name."""
        return self.configuration["label"].lower()

    @property
    def expression_name(self) -> str | None:
        """Return expression column name from the configuration file."""
        return self.processing_configuration.get("expression_name", None).lower()

    @property
    def expression_column_name(self) -> str:
        """Return the used expression column name from the configuration file."""
        return self.processing_configuration.get("expression_column_name", "expression").lower()

    @property
    def excluded_features(self) -> list[str]:
        """Return the excluded features list."""
        return [
            feature.lower() for feature in self.processing_configuration.get("exclude_features", [])
        ]

    @property
    def float_features(self) -> list[str]:
        """Return float features list."""
        return [feature.lower() for feature in self.features_configuration.get("float", [])]

    @property
    def int_features(self) -> list[str]:
        """Return int features list."""
        return [feature.lower() for feature in self.features_configuration.get("int", [])]

    @property
    def categorical_features(self) -> list[str]:
        """Return categorical features list."""
        return [feature.lower() for feature in self.features_configuration.get("categorical", [])]

    @property
    def ids_columns(self) -> list[str]:
        """Return ids columns list."""
        ids = [id_.lower() for id_ in self.features_configuration.get("ids", [])]
        if self.id_column not in ids:
            ids.append(self.id_column)
        return ids

    @property
    def id_column(self) -> str:
        """Return main id column list."""
        return self.features_configuration["id"].lower()

    @property
    def processed_data_path(self) -> Path:
        """Return processed data path."""
        return self.experiment_proc_data_folder / f"{self.data_path.stem}.csv"

    @property
    def featureizer_path(self) -> Path:
        """Return featureizer path."""
        return self.experiment_proc_data_folder / "featureizer.pkl"

    @property
    def seed(self) -> int:
        """Return seed."""
        return self.processing_configuration.get("seed", 0)

    @property
    def fold(self) -> int:
        """Return nbr of fold."""
        return self.processing_configuration.get("fold", 5)

    @property
    def validation_splits_path(self) -> Path:
        """Return validation split path."""
        return (
            self.experiment_proc_data_folder
            / f"validation_split_seed:{self.seed}_fold:{self.fold}.csv"
        )

    @property
    def features_configuration_path(self) -> Path:
        """Return the path where the feature configuration saved will be saved."""
        return self.experiment_proc_data_folder / "features_configuration.yml"

    @property
    def remove_unnecessary_folders(self) -> bool:
        """Return remove_unnecessary_folders bool variable."""
        return self.processing_configuration.get("remove_unnecessary_folders", False)

    @property
    def nan_ratio(self) -> float:
        """Return nan ratio."""
        return self.processing_configuration.get("nan_ratio", 0.6)

    @property
    def experiments(self) -> dict[str, Any]:
        """Return experiments configuration."""
        return self.configuration["experiments"]

    def force_validation_strategy(self) -> None:
        """Force use validation strategy to True."""
        self.use_validation_strategy = True

    def load_features_configuration(
        self, click_ctx: click.core.Context | Any, force_gcp: bool
    ) -> dict[str, Any]:
        """Load features configuration from different sources.

        If the file is not saved in the checkpoint directory and it's train mode it will be loaded
        from features folder
        """
        if self.features_configuration_path.exists():
            return load_yml(self.features_configuration_path)
        if self.is_train:

            features_configuration = self.load_features_configuration_from_source(
                click_ctx, force_gcp
            )

            self.save_features_configuration(features_configuration)
            return features_configuration
        raise FileNotFoundError("Features configuration not found.")

    def load_features_configuration_from_source(
        self, click_ctx: click.core.Context | Any, force: bool
    ) -> dict[str, Any]:
        """Return features configuration."""
        trainable_features_name = self.processing_configuration.get("trainable_features", None)
        if trainable_features_name:
            file_path = self.maybe_pull_features_configuration_from_gcp(
                click_ctx, trainable_features_name, force
            )
            if file_path.exists():
                return load_yml(file_path)

            raise FileNotFoundError(
                f"{file_path.name} not found in {self.data_path.parent} folder "
                "please create yml file contains"
                "the list of the ids,floats,int,categorical features."
                "or pull it from Biondeep-ig bucket"
            )

        raise KeyError("trainable_features key should be defined in the processing section")

    def maybe_pull_features_configuration_from_gcp(
        self, click_ctx: click.core.Context | Any, file_path: str, force: bool
    ) -> Path:
        """Pull the features configuration file  from the bucket.

        if the path starts with gs:// and return the local path,
        or return the same path if it's a local path.
        """
        if not file_path.startswith(GS_BUCKET_PREFIX):
            return self.data_path.parent / file_path
        local_path = str(self.data_path.parent / remove_bucket_prefix(file_path, -1))
        if (not Path(local_path).exists()) or force:
            log.info("Download features configuration from GCP: %s", file_path)
            click_ctx.invoke(pull, bucket_path=file_path, local_path=local_path, force=force)
        return Path(local_path)

    def load_data(self) -> Dataset:
        """Return processed data whatever is it the saved or the processed from the beginning."""
        self.process()

        if self.is_train:
            self.validation_splits()
        self.save_processed_data()

        return self

    def find_features(self, all_features: list[str], index: str) -> list[str]:
        """This method generates a features list from input index."""
        return list(
            filter(
                lambda feature: re.findall(index, feature),
                all_features,
            )
        )

    def get_feature_list(self) -> list[str]:
        """This function returns the final features list.

        It runs feature exclusion based on a prefix/suffix, a specific pattern or full feature name.
        """
        self.excluded_features_list = []

        for pattern in self.excluded_features:
            keyword_exclude_features = self.find_features(self.all_features, pattern)
            self.excluded_features_list += keyword_exclude_features
            log.info("Excluding %s features %s", pattern, len(keyword_exclude_features))

        if len(self.excluded_features_list) > 0:

            log.info("Done excluding %s features", len(self.excluded_features_list))
        return [
            feature for feature in self.all_features if feature not in self.excluded_features_list
        ]

    def process(self) -> None:
        """Load and process data."""
        log.info("load data set from %s", self.data_path)
        self.data = read_data(str(self.data_path))
        self.features_lower_case()
        if self.is_inference:
            self.features = [
                feature for feature in self.all_features if feature in self.data.columns
            ]
        else:
            self.features = self.get_feature_list()

        if self.process_label:
            self.clean_target()

        self.check_features_matching()

        if self.is_train:
            log.debug("Train mode featureizer is created")
            self.featureizer = Featureizer(
                processing_configuration=self.processing_configuration,
                float_features=self.float_features,
                int_features=self.int_features,
                categorical_features=self.categorical_features,
            )
        else:
            if self.featureizer_path.exists():
                self.featureizer = load_pkl(self.featureizer_path)
                log.debug("Test mode featureizer is loaded")
            else:
                raise Exception("featureizer is not fitted")

        self.data = self.featureizer.fillna(self.data)
        self.data = self.featureizer.label_encoder(self.data)
        self.data = self.featureizer.normalize(self.data)
        self.data = self.featureizer.normalize_per_id(self.data)

        if not self.featureizer_path.exists():
            self.save_featureizer()

    def maybe_pull_data_from_gcp(
        self, file_path: str, click_ctx: click.core.Context | Any, force_gcp: bool
    ) -> Path:
        """Pull the input dataset from the bucket.

        if the path starts with gs:// and return the local path,
        or return the same path if it's a local path.
        """
        if (not file_path.startswith(GS_BUCKET_PREFIX)) and Path(file_path).exists():
            return Path(file_path)
        if file_path.startswith(GS_BUCKET_PREFIX):
            local_path = str(DATA_DIRECTORY / remove_bucket_prefix(file_path))
            if (not Path(local_path).exists()) or force_gcp:
                log.info("Download data from GCP: %s", file_path)
                click_ctx.invoke(
                    pull, bucket_path=file_path, local_path=local_path, force=force_gcp
                )
            return Path(local_path)

        raise FileNotFoundError("The file is not available either locally or in GCP.")

    def features_lower_case(self) -> None:
        """Make data columns in lowercase format."""
        self.data.columns = [col.lower() for col in self.data.columns]

    def check_features_matching(self) -> None:
        """Check if the data has all the needed features."""
        features_missing = [
            feature for feature in self.features if feature not in self.data.columns.tolist()
        ]
        if len(features_missing):
            raise KeyError(f"data is missing the following features: {' '.join(features_missing)}")

    def clean_target(self) -> None:
        """Clean Label columns by Keeping only 0 and 1 values."""
        if self.label in self.data.columns:
            if self.data[self.label].isna().mean() >= self.nan_ratio:
                raise ValueError(
                    f"the input data {self.data_path} has a {self.label} column but it's Nan",
                )
            log.info("data has %.2f of missing label.", self.data[self.label].isna().mean())
            self.data = self.data[self.data[self.label].isin([0, 1, "1", "0", "0.0", "1.0"])]
            self.data[self.label] = self.data[self.label].astype(int)
        else:
            raise KeyError(f"the input data  {self.data_path} is without {self.label} column.")

    def save_processed_data(self) -> None:
        """Save processed data."""
        columns = self.ids_columns + self.features + self.splits_columns

        if self.label in self.data.columns:
            columns = columns + [self.label]

        self.data[columns].to_csv(self.processed_data_path, index=False)
        log.info("save processed data to %s", self.processed_data_path)

    def save_featureizer(self) -> None:
        """Load the saved featureizer."""
        save_as_pkl(self.featureizer, self.featureizer_path)
        log.info("save featureizer  to %s", self.featureizer_path)

    def validation_splits(self) -> None:
        """Apply cross validations strategy  for each specif experiment."""
        # Single Model
        kfold_exps = list(set(self.experiments.keys()) & set(KFOLD_EXP_NAMES))
        # Kfold Experiment
        if kfold_exps:
            kfold_split_name = self.experiments[kfold_exps[0]]["split_column"]
            self.splits_columns.append(kfold_split_name)
            self._kfold_split(kfold_split_name, self.seed)

        if SINGLE_MODEL_NAME in self.experiments.keys():

            single_model_split_name = self.experiments[SINGLE_MODEL_NAME]["validation_column"]
            self.splits_columns.append(single_model_split_name)
            self._single_model_split(single_model_split_name)

        try:
            self.data[self.ids_columns + self.splits_columns].to_csv(
                self.validation_splits_path, index=False
            )
        except KeyError as err:
            raise KeyError(
                f" [{', '.join(self.splits_columns)}]  columns are not defined"
                + " in the provided train data : Check the name of these columns or"
                + " set validation strategy to True."
            ) from err

    def _single_model_split(self, split_name: str) -> None:
        """Perform the splitting for Single Model and check the split column."""
        if self.use_validation_strategy:
            self.train_val_split(split_name)
        else:
            if split_name not in self.data.columns:
                message = (
                    f"Split column : {split_name} is not "
                    + " defined in the train data. Please set validation_strategy to True"
                    + f" or rename the split column to {split_name}"
                )
                raise ValueError(message)

    def _kfold_split(self, split_column: str, seed: int) -> None:
        """Perform the kfold splitting and check the split column."""
        if self.use_validation_strategy:
            if self.fold:

                self.kfold_split(split_column, seed)
            else:
                message = (
                    "Please provide a fold value to split the dataset !"
                    + "You can add an attribute in the configuration file under processing section"
                    + " as following : fold: number of folds"
                )
                raise Warning(message)
        else:
            if split_column not in self.data.columns:
                message = (
                    f"Split column : {split_column} is not"
                    + " defined in the train data. Please set validation_strategy to True"
                    + f" or rename the split column to {split_column}"
                )
                raise ValueError(message)
            data_fold = self.data[split_column].nunique()
            if data_fold != self.fold:
                message = (
                    "The number of dataset splits is different from the number specified"
                    + " in the configuration file ! Please set validation_strategy to True "
                    + "or use the right number of splits\n\n dataset:"
                    + f"{data_fold}\n configuration: {self.fold}"
                )
                raise ValueError(message)

    def train_val_split(self, split_column: str) -> None:
        """Split data into train and val set."""
        self.data.reset_index(inplace=True, drop=True)

        _, validation_index = train_test_split(
            self.data.index,
            test_size=self.processing_configuration["validation_ratio"],
            random_state=self.seed,
        )
        self.data[split_column] = 0
        self.data.loc[validation_index, split_column] = 1

    def kfold_split(self, split_column: str, random_state: int) -> None:
        """Split data into k folds."""
        self.data.reset_index(inplace=True, drop=True)
        kfold = KFold(n_splits=self.fold, shuffle=True, random_state=random_state).split(
            self.data.index
        )
        self.data[split_column] = 0
        for i, (_, validation_index) in enumerate(kfold):
            self.data.loc[validation_index, split_column] = int(i)

    def save_features_configuration(self, features_configuration: dict[str, Any]) -> None:
        """Save features configuration."""
        save_yml(features_configuration, self.features_configuration_path)

    def __call__(self, kind: str = "all") -> pd.DataFrame:
        """Return processed data with the needed features using the argument kind."""
        if kind == "all":
            return self.data
        if kind == "features":
            return self.data[self.features]
        if kind == "ids":
            return self.data[self.ids_columns]
        if kind == "label":
            return self.data[self.label]

        raise KeyError("{kind} is  not defined")


class Featureizer:
    """Handle normalization, transform_to_numeric and fillna method."""

    def __init__(
        self,
        processing_configuration: dict[str, Any],
        float_features: list[str],
        int_features: list[str],
        categorical_features: list[str],
    ) -> None:
        """Init Method."""
        self.processing_configuration = processing_configuration
        self.float_features = float_features
        self.int_features = int_features
        self.categorical_features = categorical_features
        self.label_encoder_fitted = False
        self.normalizer_fitted = False

        self.normalizer_name = processing_configuration.get("normalizer", None)
        self.normalizer_per_id_name = processing_configuration.get("normalizer_per_id", None)
        self.normalizer_id = processing_configuration.get("normalizer_id", None)

        self.normalizer = None
        self.normalizer_per_id = self._get_normalizer(self.normalizer_per_id_name)

        self.label_encoder_map: dict[str, dict[str, str]] = {}
        self.transform_to_numeric_fitted = True  # is set it to true because for the current version no need for transform_to_numeric
        self.fillna_method = processing_configuration.get("fill_nan_method", "keep")
        self.bool_features: list[str] = []
        self.features: list[str] = []
        if (self.normalizer_name is not None) and (self.normalizer_per_id_name is not None):
            raise ValueError("normalizer and normalizer_per_id cannot be used at the same time")

    def transform_to_numeric(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa
        """Transform features to numeric."""

        if not self.transform_to_numeric_fitted:
            log.debug("--- categorical_features : %s", len(self.get_categorical_features(data)))
            for col in self.get_categorical_features(data):
                data[col] = data[col].apply(lambda x: str(x).replace("\t", "").replace("\n", ""))
                data[col] = data[col].replace({"n.d.": np.nan, "NE": np.nan, "?": np.nan})
                try:
                    data[col] = data[col].astype(float)
                    log.debug("    *Feature %s could be transformed to numeric.", col)
                except ValueError:
                    log.debug("    *Feature %s couldn't be transformed to numeric.", col)

            self.int_features = self.get_int_features(data)
            self.categorical_features = self.get_categorical_features(data)
            self.bool_features = self.get_bool_features(data)
            self.float_features = self.get_float_features(data)
            self.transform_to_numeric_fitted = True
            log.debug("--- categorical_features : %s", len(self.categorical_features))
        else:
            for col in self.float_features:

                try:
                    data[col] = data[col].astype(float)
                except ValueError:
                    try:
                        data[col] = data[col].apply(
                            lambda x: str(x).replace("\t", "").replace("\n", "")
                        )
                        data[col] = data[col].replace({"n.d.": np.nan, "NE": np.nan, "?": np.nan})
                        data[col] = data[col].astype(float)
                    except ValueError as v:
                        raise ValueError(
                            "{col} could be transformed to float as in train set."
                        ) from v

        return data

    def label_encoder(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        if not self.transform_to_numeric_fitted:
            raise Exception("Transform_to_numeric has to be called before label_encoder.")

        if not self.label_encoder_fitted:
            log.debug("label encoder fit")
            for col in self.categorical_features + self.bool_features:

                self.label_encoder_map[col] = {}
                unique_values = data[col].dropna().unique()
                for value, key in enumerate(unique_values):
                    self.label_encoder_map[col][key] = str(value)
            self.label_encoder_fitted = True

        if len(self.categorical_features + self.bool_features):
            log.debug("Label encoding")
        for col in self.categorical_features + self.bool_features:

            data[col] = data[col].map(self.label_encoder_map[col])

        return data

    def _get_normalizer(self, normalizer_name: str) -> Any:
        """Returns a normalizer based on the specified normalizer name."""
        normalizers = {
            "StandardScaler": preprocessing.StandardScaler(),  # Standardizes features by removing the mean and scaling to unit variance.
            "MinMaxScaler": preprocessing.MinMaxScaler(
                feature_range=(0, 1)
            ),  # Transforms features by scaling each feature to a given range.
            "QuantileTransformer": preprocessing.QuantileTransformer(
                output_distribution="normal"
            ),  # Transforms features to follow a normal distribution.
            "PowerTransformer": preprocessing.PowerTransformer(),  # Applies power transformations to make the data more Gaussian-like.
            "FunctionTransformer": preprocessing.FunctionTransformer(  # Applies a specified function to the input data.
                np.arcsinh, inverse_func=np.sinh, validate=False, check_inverse=True
            ),
        }

        if normalizer_name:
            # get normalizer from normalizers dict if normalizer name is not in the dict return None
            normalizer = normalizers.get(normalizer_name, None)
            if normalizer is None:
                raise ValueError(
                    f"Normalizer: {normalizer_name} not aviable, "
                    f"Available methods: {', '.join(normalizers.keys())}"
                )
            return normalizer
        return None

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the float features in the data using the normalizer."""
        if not self.transform_to_numeric_fitted:
            raise Exception("Transform_to_numeric has to be called before label_encoder.")

        if self.normalizer_name and not self.normalizer_fitted:
            self.normalizer = self._get_normalizer(self.normalizer_name)

        if self.normalizer:
            if not self.normalizer_fitted:
                self.normalizer = self.normalizer.fit(data[self.float_features])
                self.normalizer_fitted = True

            data[self.float_features] = self.normalizer.transform(data[self.float_features])

        return data

    def _apply_normalization(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to the input data."""
        normalized_data = self.normalizer_per_id.fit_transform(x[self.float_features].values)
        x[self.float_features] = normalized_data
        return x

    def normalize_per_id(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the given data per patient ID."""
        if self.normalizer_per_id:
            if self.normalizer_id:
                data = data.groupby(self.normalizer_id).apply(self._apply_normalization)
            else:
                raise ValueError("normalize_id is None")
        return data

    def fillna(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values."""
        if not self.transform_to_numeric_fitted:
            raise Exception("Transform_to_numeric has to be called before label_encoder.")

        if self.fillna_method == "mean":
            log.debug("Fillna method: mean")
            for col in self.float_features:
                data[col] = data[col].fillna(data[col].mean())
            for col in self.int_features:
                data[col] = data[col].fillna(round(data[col].mean()))
            for col in self.categorical_features:
                data[col] = data[col].fillna("Nan")
            return data
        if self.fillna_method == "keep":
            log.debug("Fillna method: keep")
            return data

        raise ValueError(
            f"Fillna method: {self.fillna_method} not known; available methods: mean, keep"
        )

    def get_categorical_features(self, data: pd.DataFrame) -> list[str]:
        """Return categorical features."""
        return data[self.features].select_dtypes("object").columns.tolist()

    def get_int_features(self, data: pd.DataFrame) -> list[str]:
        """Return numerical features."""
        return data[self.features].select_dtypes(["float", "int"]).columns.tolist()

    def get_bool_features(self, data: pd.DataFrame) -> list[str]:
        """Return bool features."""
        return data[self.features].select_dtypes("bool").columns.tolist()

    def get_float_features(self, data: pd.DataFrame) -> list[str]:
        """Return float features."""
        return data[self.features].select_dtypes(["float"]).columns.tolist()
