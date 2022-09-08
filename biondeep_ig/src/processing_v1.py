"""Holds functions for data preprocessing."""
from collections import defaultdict
from pathlib import Path
from xml.dom import NotFoundErr

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from biondeep_ig import DATAPROC_DIRACTORY
from biondeep_ig import DKFOLD_MODEL_NAME
from biondeep_ig import FEATURES_DIRECTORY
from biondeep_ig import KFOLD_MODEL_NAME
from biondeep_ig import SINGLE_MODEL_NAME
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.utils import load_pkl
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import read_data
from biondeep_ig.src.utils import save_as_pkl
from biondeep_ig.src.utils import save_yml

log = get_logger("Processing")


class Dataset:
    """Class to handle data processing and cleaning."""

    # ----------------------------------------------
    def __init__(self, data_path, configuration, is_train, experiment_path, is_unlabeled=False):
        """Initialize the Dataset class."""
        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.is_train = is_train
        self.experiment_path = experiment_path
        self.is_unlabeled = is_unlabeled
        self.configuration = configuration

        self.experiment_proc_data_folder = self.experiment_path / DATAPROC_DIRACTORY
        self.experiment_proc_data_folder.mkdir(exist_ok=True, parents=True)
        self.features_configuration = self.load_features_configuration()
        self.splits_columns = []

    @property
    def processing_configuration(self):
        """Return processing configuration."""
        return self.configuration["processing"]

    @property
    def label(self):
        """Return label name."""
        return self.configuration["label"].lower()

    @property
    def expression_name(self):
        """Return expression column name from the configuration file."""
        return self.processing_configuration.get("expression_name", None)

    @property
    def expression_column_name(self):
        """Return the used expression column name from the configuration file."""
        return self.processing_configuration.get("expression_column_name", "expression")

    @property
    def excluded_features(self):
        """Return the excluded features list."""
        return [
            feature.lower() for feature in self.processing_configuration.get("exclude_features", [])
        ]

    @property
    def float_features(self):
        """Return float features list."""
        # return [
        #     feature.lower()
        #     for feature in self.features_configuration.get("float", [])
        #     if feature.lower() in self.data.columns
        # ]
        return [feature.lower() for feature in self.features_configuration.get("float", [])]

    @property
    def int_features(self):
        """Return int features list."""
        # return [
        #     feature.lower()
        #     for feature in self.features_configuration.get("int", [])
        #     if feature.lower() in self.data.columns
        # ]
        return [feature.lower() for feature in self.features_configuration.get("int", [])]

    @property
    def categorical_features(self):
        """Return categorical features list."""
        # return [
        #     feature.lower()
        #     for feature in self.features_configuration.get("categorical", [])
        #     if feature.lower() in self.data.columns
        # ]
        return [feature.lower() for feature in self.features_configuration.get("categorical", [])]

    @property
    def ids_columns(self):
        """Return ids columns list."""
        ids = [id_.lower() for id_ in self.features_configuration.get("ids", [])]
        if self.id_column not in ids:
            ids.append(self.id_column)
        return ids

    @property
    def id_column(self):
        """Return main id column list."""
        return self.features_configuration["id"].lower()

    @property
    def features(self):
        """Return all features."""
        all_features = self.float_features + self.int_features + self.categorical_features
        return [feature for feature in all_features if feature not in self.excluded_features]

    @property
    def processed_data_path(self):
        """Return processed data path."""
        return self.experiment_proc_data_folder / f"{self.data_path.stem}.csv"

    @property
    def featureizer_path(self):
        """Return featureizer path."""
        return self.experiment_proc_data_folder / "featureizer.pkl"

    @property
    def seed(self):
        """Return seed."""
        return self.processing_configuration.get("seed", 0)

    @property
    def fold(self):
        """Return nbr of fold."""
        return self.processing_configuration.get("fold", 5)

    @property
    def validation_splits_path(self):
        """Return validation split path."""
        return (
            self.experiment_proc_data_folder
            / f"validation_split_seed:{self.seed}_fold:{self.fold}.csv"
        )

    @property
    def features_configuration_path(self):
        """Return the path where the feature configuration saved will be saved."""
        return self.experiment_proc_data_folder / "features_configuration.yml"

    @property
    def remove_proc_data(self):
        """Return remove proc data bool variable."""
        return self.processing_configuration.get("remove_proc_data", False)

    @property
    def use_validation_strategy(self):
        """Return validation strategy bool varibale."""
        return self.processing_configuration.get("validation_strategy", True)

    @property
    def nan_ratio(self):
        """Return nan ratio."""
        return self.processing_configuration.get("nan_ratio", 0.6)

    @property
    def process_data(self):
        """Return  process_data flag."""
        return self.processing_configuration.get("process_data", True)

    def load_features_configuration(self):
        """Load features configuration from different sources.

        If the file is not saved in the checkpoint direceotry and it's train mode it will be loaded
        from features folder
        """
        if self.features_configuration_path.exists():
            return load_yml(self.features_configuration_path)

        if self.is_train:
            features_configuration = self.load_features_configuration_from_source()
            self.save_features_configuration(features_configuration)
            return features_configuration
        print(self.features_configuration_path)
        raise FileNotFoundError("Features configuration not found.")

    def load_features_configuration_from_source(self):
        """Return features configuration."""
        trainable_features_name = self.processing_configuration.get("trainable_features", None)
        if trainable_features_name:
            file_path = FEATURES_DIRECTORY / f"{trainable_features_name}.yml"
            if file_path.exists():
                return load_yml(file_path)

            raise FileNotFoundError(
                (
                    f"{trainable_features_name}.yml not found in {FEATURES_DIRECTORY} ",
                    "please create yml file contains",
                    "the list of the ids,floats,int,categorical features.",
                )
            )

        raise KeyError("trainable_features key should be defined in the processing section")

    def load_data(self):
        """Return processed data whatever is it the saved or the processed from the begining."""
        if self.processed_data_path.exists():
            log.info(f"load porcessed data from {self.processed_data_path}")
            self.data = read_data(str(self.processed_data_path))
        else:
            self.process()
            self.save_processed_data()

        if self.is_train:
            self.validation_splits()
        return self

    def process(self):
        """Load and process data."""
        log.info(f"load data set from {self.data_path}")
        self.data = read_data(str(self.data_path))
        self.features_lower_case()
        if self.process_data:
            self.clean_target()
            self.rename_expression_column()
            self.check_features_matching()
            self.replace_missing_values_by_nan()
            self.check_data_qulaity()

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

        if not self.featureizer_path.exists():
            self.save_featureizer()

    def features_lower_case(self):
        """Make data columns in lowercase format."""
        self.data.columns = [col.lower() for col in self.data.columns]

    def check_features_matching(self):
        """Check if teh data has all the needed features."""
        features_missing = [
            feature for feature in self.features if feature not in self.data.columns.tolist()
        ]
        if len(features_missing):
            raise KeyError(f"data is missing the follwoing features: {' '.join(features_missing)}")

    def clean_target(self):
        """Clean Label columns by Keeping only 0 and 1 values."""
        self.data = self.data[self.data[self.label].isin([0, 1, "1", "0", "0.0", "1.0"])]
        self.data[self.label] = self.data[self.label].astype(int)

    def rename_expression_column(self):
        """Rename real expression column name to expression."""
        if self.expression_column_name in self.features:
            if self.expression_name:
                if self.expression_name in self.data.columns:
                    self.data.rename(
                        columns={self.expression_name: self.expression_column_name}, inplace=True
                    )
                else:
                    raise KeyError(
                        (
                            f"{self.expression_name}: does not exist in the given data please",
                            "check the expression name in the configuration file.",
                        )
                    )
            else:
                raise NotFoundErr(
                    (
                        "expression feature is available in the list of ",
                        "features but it's not defined in the configuration file.",
                    )
                )

    def replace_missing_values_by_nan(self):
        """Replace unknown characters with NaN."""
        self.data = self.data.replace("\n", "", regex=True)
        self.data = self.data.replace("\t", "", regex=True)
        nan_strings = ["NA", "NaN", "NAN", "NE", "n.d.", "nan", "na", "?", ""]
        for col in self.data.columns:
            for nanstr in nan_strings:
                self.data[col] = self.data[col].replace(nanstr, np.nan)

    def check_data_qulaity(self):
        """Check data qulaity by checking the ratio of the missing values and the duplicated column."""
        # check nan perctange.
        log.info("Check Nan ratio")
        nan_percentage = self.data[self.features].isna().mean()
        if (nan_percentage > self.nan_ratio).sum():
            log.warning(
                (
                    f" {','.join(nan_percentage[nan_percentage>self.nan_ratio].index)}",
                    " have more than 60% of nan values.",
                )
            )
        # Check duplicated columns
        log.info("Check duplicated columns by name")
        unique_columns = self.data[self.features].columns.value_counts()
        duplicate_to_drop = unique_columns[unique_columns > 1].index.tolist()
        if len(duplicate_to_drop):
            log.warning(f" {','.join(duplicate_to_drop)} are duplicated by name")

        # Check duplicated columns by value
        log.info("Check duplicated columns by values")
        all_duplicated_columns = []
        duplicate_column_names = defaultdict(list)
        for pos, column in enumerate(self.features):
            if column not in all_duplicated_columns:
                column_value = self.data[column]
                for other_column in self.features[pos + 1 :]:
                    other_column_value = self.data[other_column]
                    if column_value.equals(other_column_value):
                        duplicate_column_names[column].append(other_column)
                        all_duplicated_columns.append(other_column)
        if len(all_duplicated_columns) > 0:
            log.warning("Duplicated columns found:")
            for key, values in duplicate_column_names.items():
                log.warning(f"{key} : {','.join(values)}")

    def save_processed_data(self):
        """Save processed data."""
        columns = self.ids_columns + [self.label] + self.features + self.splits_columns
        self.data[columns].to_csv(self.processed_data_path, index=False)
        log.info(f"save processed data to {self.processed_data_path}")

    def save_featureizer(self):
        """Load the saved featureizer."""
        save_as_pkl(self.featureizer, self.featureizer_path)
        log.info(f"save featureizer  to {self.featureizer_path}")

    def validation_splits(self):
        """Apply cross validations strategy  for each specif experiment."""
        experiments = self.configuration.get("experiments")
        if SINGLE_MODEL_NAME in experiments.keys():
            single_model_split_name = experiments[SINGLE_MODEL_NAME]["validation_column"]
            self.splits_columns.append(single_model_split_name)
            if self.use_validation_strategy:
                self.train_val_split(single_model_split_name)
        if KFOLD_MODEL_NAME in experiments.keys() or DKFOLD_MODEL_NAME in experiments.keys():
            try:
                kfold_split_name = experiments[KFOLD_MODEL_NAME]["split_column"]
            except KeyError:
                kfold_split_name = experiments[DKFOLD_MODEL_NAME]["split_column"]
            self.splits_columns.append(kfold_split_name)
            if self.use_validation_strategy:
                self.kfold_split(kfold_split_name)
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

    def train_val_split(self, split_column):
        """Split data into train and val set."""
        self.data.reset_index(inplace=True, drop=True)

        _, validation_index = train_test_split(
            self.data.index,
            test_size=self.processing_configuration["validation_ratio"],
            random_state=self.seed,
        )
        self.data[split_column] = 0
        self.data.loc[validation_index, split_column] = 1

    def kfold_split(self, split_column):
        """Split data into k folds."""
        self.data.reset_index(inplace=True, drop=True)
        kfold = KFold(n_splits=self.fold, shuffle=True, random_state=self.seed).split(
            self.data.index
        )
        self.data[split_column] = 0
        for i, (_, validation_index) in enumerate(kfold):
            self.data.loc[validation_index, split_column] = int(i)

    def save_features_configuration(self, features_configuration):
        """Save features configuration."""
        save_yml(features_configuration, self.features_configuration_path)

    def __call__(self, kind="all"):
        """Return processed data with the needed features using the argument kind."""
        if kind == "all":
            return self.data
        if kind == "features":
            return self.data[self.features]
        if kind == "ids":
            return self.data[self.ids_columns]
        if kind == "label":
            return self.data[self.label]

        raise KeyError(f"{kind} is  not defined")


class Featureizer:
    """Handle normalization, transform_to_numeric and fillna method."""

    def __init__(
        self, processing_configuration, float_features, int_features, categorical_features
    ):
        """Init Method."""
        self.processing_configuration = processing_configuration
        self.float_features = float_features
        self.int_features = int_features
        self.categorical_features = categorical_features
        self.label_encoder_fitted = False
        self.normalizer_fitted = False
        self.normalize_name = processing_configuration.get("normalizer", None)
        self.label_encoder_map = {}
        self.normalizer = None
        self.transform_to_numeric_fitted = True  # is set it to true because for the current version no need for transform_to_numeric
        self.fillna_method = processing_configuration.get("fillna_method", "keep")
        self.bool_features = []

    def transform_to_numeric(self, data):  # noqa
        """Transform features to numeric."""

        if not self.transform_to_numeric_fitted:
            log.debug(f"--- categorical_features : {len(self.get_categorical_features(data))}")
            for col in self.get_categorical_features(data):
                data[col] = data[col].apply(lambda x: str(x).replace("\t", "").replace("\n", ""))
                data[col] = data[col].replace({"n.d.": np.nan, "NE": np.nan, "?": np.nan})
                try:
                    data[col] = data[col].astype(float)
                    log.debug(f"    *Feature {col} could be transformed to numeric.")
                except ValueError:
                    log.debug(f"    *Feature {col} couldn't be transformed to numeric.")

            self.int_features = self.get_int_features(data)
            self.categorical_features = self.get_categorical_features(data)
            self.bool_features = self.get_bool_features(data)
            self.float_features = self.get_float_features(data)
            self.transform_to_numeric_fitted = True
            log.debug(f"--- categorical_features : {len(self.categorical_features)}")
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
                    except ValueError:
                        raise ValueError(f"{col} could be transformed to float as in train set.")

        return data

    def label_encoder(self, data):
        """Encode categorical features."""
        if not self.transform_to_numeric_fitted:
            raise Exception("Transform_to_numeric has to be called before label_encoder.")

        if not self.label_encoder_fitted:
            log.debug("label encoder fit")
            for col in self.categorical_features + self.bool_features:

                self.label_encoder_map[col] = {}
                unique_values = data[col].dropna().unique()
                for value, key in enumerate(unique_values):
                    self.label_encoder_map[col][key] = value
            self.label_encoder_fitted = True

        if len(self.categorical_features + self.bool_features):
            log.debug("Label encoding")
        for col in self.categorical_features + self.bool_features:

            data[col] = data[col].map(self.label_encoder_map[col])

        return data

    def normalize(self, data):
        """Normalize float features."""
        if not self.transform_to_numeric_fitted:
            raise Exception("Transform_to_numeric has to be called before label_encoder.")

        if not self.normalizer_fitted:
            if self.normalize_name:
                if self.normalize_name == "UnitVariance":
                    log.debug(f"{self.normalize_name} is used as Normalizer")
                    self.normalizer = preprocessing.StandardScaler().fit(data[self.float_features])
                elif self.normalize_name == "Scale01":
                    self.normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(
                        data[self.float_features]
                    )
                    log.debug(f"{self.normalize_name} is used as Normalizer")
                else:
                    raise ValueError(
                        (
                            f"Normalizer : {self.normalize_name} not known "
                            "; available methods: UnitVariance, Scale01 "
                        )
                    )
            self.normalizer_fitted = True
        if self.normalizer:
            log.debug("Normalization")
            data[self.float_features] = self.normalizer.transform(data[self.float_features])
        return data

    def fillna(self, data):
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
