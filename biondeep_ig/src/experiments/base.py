"""File that contains the implementation of the base classes."""
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import biondeep_ig.src.models as src_model
from biondeep_ig.src import Evals
from biondeep_ig.src import FEATURES_DIRECTORY
from biondeep_ig.src import MODELS_DIRECTORY
from biondeep_ig.src import SINGLE_MODEL_NAME
from biondeep_ig.src.evaluation import Evaluation
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.processing import Dataset
from biondeep_ig.src.utils import get_model_by_name
from biondeep_ig.src.utils import load_features
from biondeep_ig.src.utils import plotting_kfold_shap_values
from biondeep_ig.src.utils import plotting_shap_values
from biondeep_ig.src.utils import save_features
from biondeep_ig.src.utils import save_yml

log = get_logger("Train/Exp")


class BaseExperiment(ABC):
    """Base experiment class implementation.

    Attributes:
        train_data_path: path of the train data
        test_data_path: path of the test data
        configuration: the full configuration of the experiment in dictionary format
        folder_name: main folder name where the run will be saved
        experiment_name: the experiment name (KfoldExperiment,SingleModel,..)
        sub_folder_name: folder name  where the experiment with the specific
        feature list will be saved folder_name/experiment_name/sub_folder_name)
        unlabeled_path: path to the unlabeled data
        experiment_directory: path where the experiment
        will be saved in case experiment_directory is not None


    """

    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        configuration: Dict[str, Any],
        folder_name: str,
        experiment_name: Optional[str] = None,
        sub_folder_name: Optional[str] = None,
        unlabeled_path: Optional[str] = None,
        experiment_directory: Optional[Path] = None,
        features_file_path: Optional[str] = None,
    ):
        """Class init."""
        self.test_data_path = Path(test_data_path)
        self.train_data_path = Path(train_data_path) if train_data_path else None
        self.unlabeled_path = unlabeled_path

        self.configuration = configuration
        self.experiment_name = experiment_name
        self.folder_name = folder_name
        self.sub_folder_name = sub_folder_name
        self.experiment_directory = self._get_experiment_directory(experiment_directory)
        self.splits_path = MODELS_DIRECTORY / self.folder_name / "splits"
        self.checkpoint_directory = self.experiment_directory / "checkpoint"
        self.prediction_directory = self.experiment_directory / "prediction"

        self.eval_directory = self.experiment_directory / "eval"
        self.curve_plot_directory = self.eval_directory / "curve_plots"
        self.model_cls = get_model_by_name(src_model, self.model_type)
        self.features_file_path = (
            features_file_path
            if features_file_path
            else self.features_directory / self.configuration["features"]
        )
        self.features = [x.lower() for x in load_features(self.features_file_path)]

        self.save_model = True
        self.evaluator = Evaluation(
            label_name=self.label_name,
            eval_configuration=self.eval_configuration,
            curve_plot_directory=self.curve_plot_directory,
        )
        self.plot_shap_values: bool
        self.plot_kfold_shap_values: Optional[bool]
        self.prediction_columns_name: List[str]
        self.statistics_opts: List[str]
        self.kfold_prediction_name: Optional[List[str]]

    @property
    def model_type(self):
        """Type of the model."""
        return self.check_and_return_model_type(self.configuration["model_type"])

    @property
    def label_name(self):
        """Label name."""
        return self.configuration["label"]

    @property
    def is_checkpoint_directory_empty(self):
        """Checking if directory is empty."""
        if self.checkpoint_directory.exists():
            return not any(self.checkpoint_directory.iterdir())
        return False

    @property
    def task(self):
        """Return the task of the run."""
        if "8" in self.label_name:
            return "CD8"
        if "4" in self.label_name:
            return "CD4"
        return None

    @property
    def features_directory(self):
        """Return features list directory."""
        if self.task:
            return FEATURES_DIRECTORY / self.task
        return FEATURES_DIRECTORY

    @property
    def eval_configuration(self):
        """Return evaluation configuration."""
        return self.configuration["evaluation"]

    @property
    def model_configuration(self):
        """Model cfg."""
        return self.configuration["model"]

    @model_configuration.setter
    def model_configuration(self, model_configuration):
        self.configuration["model"] = model_configuration

    @abstractmethod
    def train(self):
        """Training method."""

    @abstractmethod
    def predict(self, save_df):
        """Predict method."""

    @abstractmethod
    def eval_exp(self):
        """Evaluation method."""

    @abstractmethod
    def inference(self, data: pd.DataFrame, save_df: bool, file_name: str = ""):
        """Inference method."""

    def initialize_checkpoint_directory(self, tuning_option: bool = False):
        """Init a ckpt directory."""
        self.experiment_directory.mkdir(exist_ok=True, parents=True)
        self.splits_path.mkdir(exist_ok=True, parents=True)
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
        if not tuning_option:
            self.experiment_directory.mkdir(exist_ok=True, parents=True)
            self.checkpoint_directory.mkdir(exist_ok=True, parents=True)
            self.prediction_directory.mkdir(exist_ok=True, parents=True)
            self.eval_directory.mkdir(exist_ok=True, parents=True)
            self.curve_plot_directory.mkdir(exist_ok=True, parents=True)
            save_features(self.features, self.experiment_directory)

    def load_data_set(self) -> None:
        """Load data set."""
        if self.train_data_path:
            self.train_data = Dataset(
                data_path=self.train_data_path,
                features=self.features,
                target=self.label_name,
                configuration=self.configuration["processing"],
                is_train=True,
                is_unlabeled=False,
                experiment_path=MODELS_DIRECTORY / self.folder_name,
            ).process_data()

        if self.unlabeled_path:
            self.unlabeled_data = Dataset(
                data_path=self.unlabeled_path,
                features=self.features,
                target=self.label_name,
                configuration=self.configuration["processing"],
                is_train=False,
                is_unlabeled=True,
                experiment_path=MODELS_DIRECTORY / self.folder_name,
            ).process_data()

            # Concatenate unlabeled to train data
            self.train_data.data = self.train_data.data.append(self.unlabeled_data.data)

        self.test_data = Dataset(
            data_path=self.test_data_path,
            features=self.features,
            target=self.label_name,
            configuration=self.configuration["processing"],
            is_train=False,
            is_unlabeled=False,
            experiment_path=MODELS_DIRECTORY / self.folder_name,
        ).process_data()

        self.test_data = self.test_data.data

    def restore(self) -> None:
        """Restoring a ckpt."""
        if self.experiment_directory.exists() & (not self.is_checkpoint_directory_empty):
            # self.configuration = load_yml(self.experiment_directory / "configuration.yml")
            self.features = load_features(self.experiment_directory / "features")
        else:
            sys.exit(
                (
                    f"Experiment with name {self.experiment_name} is not available."
                    " Please train the model first or change the save_model argument "
                    "in the model config file."
                )
            )

    # TODO replace Any with the Model class
    def create_model(self, model_path: Path, features: List[str], prediction_name: str) -> Any:
        """Creating a model.

        Args:
            model_path: path to the model
            features: list, list of features
            prediction_name: str  name of the predicted column
        Return
            model : an instance of the Models classes
        """
        return self.model_cls(
            features=features,
            parameters=self.model_configuration["model_params"],
            label_name=self.label_name,
            prediction_name=prediction_name,
            checkpoints=model_path,
            other_params=self.model_configuration["general_params"],
            folder_name=self.folder_name,
            experiment_name=self.experiment_name,
            model_type=self.model_type,
            save_model=self.save_model,
        )

    def single_fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        checkpoints: Optional[Path] = None,
        model_name: Optional[str] = None,
        prediction_name: Optional[str] = "prediction",
    ) -> Any:
        """A single fit model.

        Args:
            train: DataFrame object hold the train data
            validation: DataFrame object hold the validation data
            checkpoints: path where the model will be saved it could be None in the Tune class
            model_name: the name of the saved model
            prediction_name: the name of the predicted column
        Return:
            model: the trained model
        """
        model_path = (
            (checkpoints / model_name / "model.pkl" if model_name else checkpoints / "model.pkl")
            if checkpoints
            else None
        )
        model = self.create_model(
            model_path,
            features=self.features,
            prediction_name=prediction_name,
        )
        model.fit(train, validation)
        # model_path could be None only when the  checkpoints arg  is None and the checkpoints is None only when single_fit method is called from the Tuning class  therefore plotting_shape_values could be called only when model_path is not None
        if (self.plot_shap_values) and (model_path is not None):
            plotting_shap_values(
                model.shap_values,
                train,
                self.features,
                model_path.parent / "shap.png",
            )

        model.eval_model(data=train, data_name="train", evaluator=self.evaluator)
        model.eval_model(data=validation, data_name="validation", evaluator=self.evaluator)
        model.eval_model(data=self.test_data, data_name="test", evaluator=self.evaluator)

        return model

    # TODO Specify Any
    def multiple_fit(
        self, train: pd.DataFrame, split_column: str, sub_model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Multiple fit model.

        Retrun:
            models: list of the trained model.
        """
        models = {}
        shap_values = []
        checkpoints = (
            self.checkpoint_directory / sub_model_directory
            if sub_model_directory
            else self.checkpoint_directory
        )
        display = "##### Start Split train #####"
        for split in np.sort(train[split_column].unique()):
            if self.evaluator.print_evals:
                log.info(f"           -split {split}")
            display += f"#### split {split+1} #### "
            train_data = train[train[split_column] != split]
            validation_data = train[train[split_column] == split]
            model = self.single_fit(
                train=train_data,
                validation=validation_data,
                checkpoints=checkpoints,
                model_name=f"split_{split}",
                prediction_name=f"prediction_{split}",
            )

            display += "##### End Split train #####"
            models[split] = model
            if self.plot_shap_values:
                shap_values.append(
                    pd.DataFrame(
                        {
                            "features": self.features,
                            "scores": list(np.mean(np.abs(model.shap_values), axis=0)),
                        }
                    )
                )

        if self.plot_shap_values and self.plot_kfold_shap_values:
            shap_values_df = pd.concat(shap_values)
            shap_values_df = (
                shap_values_df.groupby("features").scores.mean().rename("scores").reset_index()
            )
            plotting_kfold_shap_values(shap_values_df, self.eval_directory / "shap.png")
        if self.evaluator.print_evals:
            log.debug(display)

        return models

    # TODO Dict
    def eval_test(self) -> Dict[str, Any]:
        """Eval method for single data set."""
        test_data = self.inference(self.test_data, file_name=self.test_data_path.stem, save_df=True)
        if self.experiment_name == SINGLE_MODEL_NAME:
            prediction_columns_name = self.prediction_columns_name
        else:
            prediction_columns_name = self.prediction_columns_name + self.kfold_prediction_name

        for prediction_name in prediction_columns_name:
            log.info(f"            {prediction_name} : ")
            self.evaluator.compute_metrics(
                data=test_data, prediction_name=prediction_name, data_name="test"
            )
        results = self._parse_metrics_to_data_frame(
            eval_metrics=self.evaluator.get_evals(), file_name=self.test_data_path.stem
        )
        fake_validation = results.copy()
        fake_validation["split"] = "validation"
        results = pd.concat([results, fake_validation])
        best_validation_scores, best_test_scores, _ = self.evaluator.get_experiment_best_scores(
            results=results,
            experiment_name=self.experiment_name,
            model_type=self.model_type,
            features_name=self.configuration["features"],
        )
        if self.experiment_name == SINGLE_MODEL_NAME:
            return {"validation": best_validation_scores, "test": best_test_scores}

        statistic_scores = self.evaluator.get_statistic_kfold_scores(
            statistic_opt=self.statistics_opts,
            prediction_columns_name=self.prediction_columns_name,
            experiment_name=self.experiment_name,
            model_type=self.model_type,
            features_name=self.configuration["features"],
        )
        return {
            "validation": best_validation_scores,
            "test": best_test_scores,
            "statistic": statistic_scores,
        }

    def plot_comparison_score(self, eval_comparison_score: str):
        """Plot comparison score  scores."""
        # TODO to be implemented

    def check_and_return_model_type(self, model_type: str) -> str:
        """Checking an returning model type."""
        if model_type not in [
            "XgboostModel",
            "LgbmModel",
            "CatBoostModel",
            "LabelPropagationModel",
            "LogisticRegressionModel",
        ]:
            raise NotImplementedError(
                (
                    f"{model_type} is not supported. Please "
                    "choose one of: [XgboostModel, LgbmModel, CatBoostModel, LabelPropagationModel]"
                )
            )
        return model_type

    def subplot1(self, test_metrics, train_metrics):
        """Plots an experiment - preprocessing."""
        dict_l = {}
        for key, val in test_metrics.items():
            if val:
                dict_l[key] = val["global"]
        df_test_metrics = pd.DataFrame(dict_l)
        df_train_metrics = pd.DataFrame(train_metrics)

        cols = [col for col in df_test_metrics.columns if "_all" not in col]
        df_test_metrics = df_test_metrics[cols]
        cols = [col for col in df_train_metrics.columns if "_all" not in col]
        df_train_metrics = df_train_metrics[cols]
        if self.comparison_score:
            df_test_metrics_cs = df_test_metrics[self.comparison_score]
            df_test_metrics = df_test_metrics.drop(columns=self.comparison_score)
            df_test_metrics = df_test_metrics.append(
                pd.DataFrame(
                    (
                        np.ones(len(df_test_metrics.columns)) * df_test_metrics_cs.loc["topk"]
                    ).reshape(1, -1),
                    columns=list(df_test_metrics),
                ),
                ignore_index=True,
            )
            df_test_metrics = df_test_metrics.append(
                pd.DataFrame(
                    (np.ones(len(df_test_metrics.columns)) * df_test_metrics_cs.loc["auc"]).reshape(
                        1, -1
                    ),
                    columns=list(df_test_metrics),
                ),
                ignore_index=True,
            )

            cs_cols = [x for x in df_train_metrics.columns if self.comparison_score in x]
            df_train_metrics_cs = df_train_metrics[cs_cols]
            df_train_metrics = df_train_metrics.drop(columns=cs_cols)
            df_train_metrics = df_train_metrics.append(
                pd.DataFrame(
                    df_train_metrics_cs.loc["topk"].values.reshape(1, -1),
                    columns=list(df_train_metrics),
                ),
                ignore_index=True,
            )
            df_train_metrics = df_train_metrics.append(
                pd.DataFrame(
                    df_train_metrics_cs.loc["auc"].values.reshape(1, -1),
                    columns=list(df_train_metrics),
                ),
                ignore_index=True,
            )

        df_test_metrics["Mean"] = df_test_metrics.mean(axis=1)
        df_train_metrics["Mean"] = df_train_metrics.mean(axis=1)

        return df_test_metrics, df_train_metrics

    def subplot2(self, fig, ax, df_cur, plot_cols, title_string, i, j):
        """Plots an experiment - performs actual plot."""
        ax[j, i] = df_cur.transpose()[plot_cols].plot.bar(
            rot=90,
            ax=ax[j, i],
            color=[np.asarray([38, 122, 55]) / 255, np.asarray([42, 203, 74]) / 255],
        )
        ax[j, i].legend(loc="lower left")
        if i == 0:
            ax[j, i].set_title(title_string[0])
        else:
            ax[j, i].set_title(title_string[1])
            # overwrite label names for SV as they should not be the same as for V (maybe do this smarter in a newer version)
            labels = [item.get_text() for item in ax[j, i].get_xticklabels()]
            labels_new = []
            for _ in range(len(labels) - 4):
                labels_new.append(self.split_column)

            k = 4
            for _ in range(4):
                labels_new.append(labels[-k])
                k = k - 1
            ax[j, i].set_xticklabels(labels_new)
        ax[j, i].set_ylim([0.0, 1.0])

        return ax, fig

    def plotting(self, test_metrics, train_metrics):
        """Plots an experiment."""
        plot_cols_topk = ["topk"]
        plot_cols_auc = ["auc"]

        df_test_metrics, df_train_metrics = self.subplot1(test_metrics, train_metrics)

        if self.comparison_score:
            df_test_metrics.index = [
                "logloss",
                "auc",
                "topk",
                "top_k_retrieval",
                "topk_CS",
                "auc_CS",
            ]
            df_train_metrics.index = [
                "logloss",
                "auc",
                "topk",
                "top_k_retrieval",
                "topk_CS",
                "auc_CS",
            ]
            plot_cols_topk = ["topk", "topk_CS"]
            plot_cols_auc = ["auc", "auc_CS"]

        iterator = 2  # two datasets: Validation/Train, Test

        fig, ax = plt.subplots(iterator, 2, figsize=(12, iterator * 6))

        for i in range(iterator):
            # posshist = 0
            if i == 0:
                df_cur = df_train_metrics

            else:
                df_cur = df_test_metrics
                # posshist = len(self.test_data.loc[self.test_data[self.label_name] == 1])
            ax, fig = self.subplot2(
                fig,
                ax,
                df_cur,
                plot_cols_topk,
                ["Global TopK per KFold (V)", "Global TopK per KFold (SV)"],
                i,
                0,
            )

            ax, fig = self.subplot2(
                fig,
                ax,
                df_cur,
                plot_cols_auc,
                ["AUC-ROC per KFold (V)", "AUC-ROC per KFold (SV)"],
                i,
                1,
            )

        fig.tight_layout()
        fig.savefig(self.eval_directory / "ScoreSummary.pdf")
        plt.close("all")

    @abstractmethod
    def _parse_metrics_to_data_frame(self, eval_metrics: Evals, file_name: str = "results"):
        """Parse Metrics results from dictionary to dataframe object."""

    def _get_experiment_directory(self, experiment_directory: Optional[Path]) -> Path:
        """Get experiment path."""
        if experiment_directory:
            return experiment_directory

        base_path = (
            MODELS_DIRECTORY / self.folder_name / self.experiment_name
            if self.experiment_name
            else MODELS_DIRECTORY / self.folder_name
        )
        return (
            base_path / self.sub_folder_name / self.model_type
            if self.sub_folder_name
            else base_path / self.model_type
        )

    def _save_prediction_name_selector(self, prediction_name_selector: str):
        """Save prediction name selector."""
        self.configuration["evaluation"]["prediction_name_selector"] = prediction_name_selector
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
