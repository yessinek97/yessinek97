"""Holds functions for data preprocessing."""
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from biondeep_ig import DATAPROC_DIRACTORY
from biondeep_ig import FEATURIZER_DIRECTORY
from biondeep_ig import ID_NAME
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.utils import load_pkl
from biondeep_ig.src.utils import save_as_pkl

log = get_logger("Processing_old")


class Datasetold:
    """Class to handle data processing and cleaning."""

    # ----------------------------------------------
    def __init__(
        self,
        data_path,
        features,
        target,
        configuration,
        is_train,
        experiment_path,
        is_unlabeled=False,
    ):
        """Initialize the Dataset class."""
        self.data_path = data_path
        self.configuration = configuration
        self.seed = configuration["seed"]

        self.is_train = is_train
        self.experiment_path = experiment_path
        self.target = target.lower()
        self.features = features
        self.nan_strings = ["NA", "NaN", "NAN", "NE", "n.d.", "nan", "na", "?"]
        self.proc_data_folder = self.experiment_path / DATAPROC_DIRACTORY
        self.proc_data_folder.mkdir(exist_ok=True, parents=True)
        self.is_unlabeled = is_unlabeled

        self.featureizer_path = (
            self.experiment_path / FEATURIZER_DIRECTORY / f"{self.hash_features}.pkl"
        )
        self.featureizer_path.parent.mkdir(exist_ok=True, parents=True)

        self.output_path = (
            self.proc_data_folder / f"{self.hash_features}_{Path(self.data_path).name}"
        ).with_suffix(".csv")

    @property
    def is_featureizer_path(self):
        """Check if featureizer path exists or not."""
        return self.featureizer_path.exists()

    @property
    def hash_features(self):
        """Return the hash of the feature list."""
        features_hash = hashlib.md5(np.sort(np.asarray(list(self.features)))).hexdigest()
        return features_hash

    @property
    def use_normalized_data(self):
        """Return use_normalized_data variable."""
        return self.configuration.get("use_normalized_data", None)

    @property
    def fill_nan_method(self):
        """Return fill_nan_method variable."""
        return self.configuration.get("fill_nan_method", None)

    def process_data(self):
        """Load and process the original dataset if not already processed."""
        if self.output_path.exists():
            self.loaddataset(str(self.output_path))
            log.debug(f"processed data loaded from  {self.output_path.name}")
            return self
        self.loaddataset(self.data_path)
        self.process()
        return self.save_get_data()

    def process(self):
        """Process Method."""
        self.features_lower_case()

        if self.is_train:

            log.debug("Train mode featureizer is created")
            self.featureizer = Featureizer(
                features=self.features,
                use_normalized_data=self.use_normalized_data,
                fillna_method=self.fill_nan_method,
            )
        else:
            if self.is_featureizer_path:
                self.featureizer = load_pkl(self.featureizer_path)
                log.debug("Test mode featureizer is loaded")
            else:
                raise Exception("featureizer is not fitted")

        self.duplicate_features()
        self.clean_utfendings()
        if self.is_unlabeled:
            self.process_unlabeled()
        self.clean_target()
        # self.remove_common_nan()
        self.data = self.featureizer.transform_to_numeric(self.data)
        self.data = self.featureizer.fillna(self.data)
        self.data = self.featureizer.label_encoder(self.data)
        self.data = self.featureizer.normalize(self.data)

        if not self.is_featureizer_path:
            save_as_pkl(self.featureizer, self.featureizer_path)

    # ----------------------------------------------
    def loaddataset(self, path):
        """Load data."""
        # Note: According to https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options, low_memory does nothing more than suppress this warning currently - a failed deprecation of pandas
        file_extension = Path(path).suffix
        if file_extension == ".csv":
            self.data = pd.read_csv(path, low_memory=False)
        elif file_extension in [".tsv", ".temp"]:
            self.data = pd.read_csv(path, low_memory=False, sep="\t")
        elif file_extension == ".xlsx":
            self.data = pd.read_excel(path)
        else:
            raise Exception("UserError: File ending of data file not known!")

    # ----------------------------------------------
    def features_lower_case(self):
        """Make all the features lowercase."""
        self.data.columns = self.data.columns.astype(str)
        self.data.columns = self.data.columns.str.lower()
        self.data[ID_NAME] = list(range(len(self.data)))

    # ----------------------------------------------
    def duplicate_features(self):
        """Remove duplicated rows."""
        pl = len(self.data.columns)
        ini_cols = self.data.columns
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        if len(self.data.columns) < pl:
            log.debug(
                (
                    f"Duplicate column names detected in {self.data_path}"
                    "and removed - maybe check your data set - will proceed"
                    "with deduplicated version "
                )
            )
            log.debug(
                f"Duplicate columns:{[x for x in ini_cols if ini_cols.to_list().count(x) > 1]}"
            )

    # ----------------------------------------------
    def clean_utfendings(self):
        """Clean the label."""
        # at first clean messed up floats:
        self.data = self.data.replace("\n", "", regex=True)
        self.data = self.data.replace("\t", "", regex=True)

        for col in self.features:
            for nanstr in self.nan_strings:
                self.data[col] = self.data[col].replace(nanstr, np.nan)

        self.data.reset_index(drop=True, inplace=True)

    # ----------------------------------------------
    def reduce_dataset(self, reduce_mode):
        """Reduce the data set for debug use."""
        self.data = self.data.iloc[0:reduce_mode]
        # ini_len = len(self.data)
        log.debug(f"Initial length = {len(self.data)}")

        self.data.reset_index(drop=True, inplace=True)

    # ----------------------------------------------
    def process_unlabeled(self):
        """Process label."""
        # in case we have unlabeled data target will not be present and we need to replace it with -1 for semisupervised sklearn libraries
        self.data[self.target] = -1.0

    # ----------------------------------------------
    def clean_target(self):
        """Cleanup the label."""
        # Treat nans & weird values in target
        for nanstr in self.nan_strings:
            self.data[self.target] = self.data[self.target].replace(nanstr, np.nan)
        self.data[self.target] = pd.to_numeric(self.data[self.target])
        self.data = self.data.loc[
            (self.data[self.target] == 0)
            | (self.data[self.target] == 1)
            | (self.data[self.target] == -1)
        ]
        self.data[self.target] = self.data[self.target].astype(int)

        self.data.reset_index(drop=True, inplace=True)

    # ----------------------------------------------
    def remove_common_nan(self):
        """Remove Common Nan."""
        # if required remove all common nans defined in the subset of features (for the actual IM runtime)
        # self.data = self.data[self.features+[self.target]]

        ini_len = len(self.data)
        log.debug(
            f"    *Target  {self.target} had " + f"{self.data[self.target].isna().sum()} Nans"
        )
        for col in self.features:
            log.debug(
                (
                    f"    *Feature {col} had {self.data[col].isna().sum()}"
                    "NaNs (after removing target NaNs)"
                )
            )
        if not self.fill_nan_method:
            self.data = self.data.dropna(subset=self.features)
            log.debug(f"Removed {ini_len - len(self.data)}  nan values")

        self.data.reset_index(drop=True, inplace=True)

    # ----------------------------------------------
    def train_val_split(self):
        """Split data into train and val set."""
        self.data.reset_index(inplace=True, drop=True)

        _, validation_index = train_test_split(
            self.data.index,
            test_size=self.configuration["validation_ratio"],
            random_state=self.seed,
        )
        self.data["validation"] = 0
        self.data.loc[validation_index, "validation"] = 1

    # ----------------------------------------------
    def kfold_split(self):
        """Split data into k folds."""
        self.data.reset_index(inplace=True, drop=True)
        kfold = KFold(
            n_splits=self.configuration["fold"], shuffle=True, random_state=self.seed
        ).split(self.data.index)
        self.data["fold"] = 0
        for i, (_, validation_index) in enumerate(kfold):
            self.data.loc[validation_index, "fold"] = int(i)

    # ----------------------------------------------
    def plot_data_histogram(self, figure_path):
        """Plot data set histogram."""
        df_plot = self.data[self.features + [self.target]]
        df_plot[self.target] = df_plot[self.target].astype(int)
        df_plot = df_plot[[self.target] + self.features]
        # features = df.columns.to_list()
        it = 0
        ix = 0
        iy = 0
        plt.figure(figsize=(6 * len(self.features), 6 * len(self.features)))
        _, axs = plt.subplots(
            int(np.ceil(np.sqrt(len(self.features)))), int(np.ceil(np.sqrt(len(self.features))))
        )
        for feature in self.features:
            if (float(it) % float(np.ceil(np.sqrt(len(self.features)))) == 0) & (it > 0):
                ix += 1
                iy = 0
            # if Regression == 0:
            a_heights, a_bins = np.histogram(
                df_plot.loc[df_plot[self.target] == 1][feature].fillna(0.0), bins=40
            )
            # this throws pandas warning -> suppress warning for this function
            b_heights, b_bins = np.histogram(
                df_plot.loc[df_plot[self.target] == 0][feature].fillna(0.0), bins=40
            )
            # else:
            #    a_heights, a_bins = np.histogram(df_plot[feature], bins=40)
            width = (a_bins[1] - a_bins[0]) / 3
            axs[ix, iy].bar(
                a_bins[:-1], a_heights, width=width, facecolor="cornflowerblue", label="positive"
            )
            # if Regression == 0:
            axs[ix, iy].bar(
                b_bins[:-1] + width, b_heights, width=width, facecolor="seagreen", label="negative"
            )
            axs[ix, iy].set_xlabel("Transformed feature value", fontsize=5)
            axs[ix, iy].set_ylabel("Count", fontsize=5)
            axs[ix, iy].set_title(feature + " (NaN = 0 or dropped)", fontsize=5)
            axs[ix, iy].legend(fontsize=4, ncol=1)
            axs[ix, iy].tick_params(axis="both", which="both", labelsize=5)
            iy += 1
            it += 1
        plt.tight_layout()
        plt.savefig(figure_path)

        df_plot[self.target] = df_plot[self.target].astype(float)

    def save_get_data(self):
        """Save data set."""
        self.data.to_csv(self.output_path, index=False)
        log.debug(f"data was saved under the name :  {self.output_path.name}")
        return self


class Featureizer:
    """Handle normalization, transform_to_numeric and fillna method."""

    def __init__(self, features, use_normalized_data, fillna_method):
        """Init Method."""
        self.features = features
        self.label_encoder_fitted = False
        self.normalizer_fitted = False
        self.use_normalized_data = use_normalized_data
        self.label_encoder_map = {}
        self.normalizer = None
        self.transform_to_numeric_fitted = False
        self.fillna_method = fillna_method
        self.categorical_features = None
        self.int_features = None
        self.bool_features = None
        self.float_features = None

    ## TODO save features with featureizer
    @property
    def is_fitted(self):
        """Check if the Featureizer is fitted or not."""
        return self.transform_to_numeric_fitted or self.label_encoder_fitted

    def get_categorical_features(self, data):
        """Return categorical features."""
        return data[self.features].select_dtypes("object").columns.tolist()

    def get_int_features(self, data):
        """Return numerical features."""
        return data[self.features].select_dtypes(["float", "int"]).columns.tolist()

    def get_bool_features(self, data):
        """Return bool features."""
        return data[self.features].select_dtypes("bool").columns.tolist()

    def get_float_features(self, data):
        """Return float features."""
        return data[self.features].select_dtypes(["float"]).columns.tolist()

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
            if self.use_normalized_data:
                if self.use_normalized_data == "UnitVariance":
                    log.debug(f"{self.use_normalized_data} is used as Normalizer")
                    self.normalizer = preprocessing.StandardScaler().fit(data[self.float_features])
                elif self.use_normalized_data == "Scale01":
                    self.normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(
                        data[self.float_features]
                    )
                    log.debug(f"{self.use_normalized_data} is used as Normalizer")
                else:
                    raise ValueError(
                        (
                            f"Normalizer : {self.use_normalized_data} not known "
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
