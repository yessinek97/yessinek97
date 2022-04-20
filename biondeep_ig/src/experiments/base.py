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
import seaborn as sns

import biondeep_ig.src.models as src_model
from biondeep_ig import Evals
from biondeep_ig import FEATURES_SELECTION_DIRACTORY
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig import SINGLE_MODEL_NAME
from biondeep_ig.src.evaluation import Evaluation
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.processing import Datasetold
from biondeep_ig.src.processing_v1 import Dataset
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
        train_data: Dataset,
        test_data: Dataset,
        configuration: Dict[str, Any],
        folder_name: str,
        experiment_name: Optional[str] = None,
        sub_folder_name: Optional[str] = None,
        unlabeled_path: Optional[str] = None,
        experiment_directory: Optional[Path] = None,
        features_file_path: Optional[str] = None,
    ):
        """Class init."""
        self.train_data = train_data
        self.test_data = test_data
        self.unlabeled_path = unlabeled_path

        self.test_data_path = test_data.data_path
        self.train_data_path = train_data.data_path if isinstance(train_data, Dataset) else None

        self.configuration = configuration
        self.experiment_name = experiment_name
        self.folder_name = folder_name
        self.sub_folder_name = sub_folder_name
        self.experiment_directory = self._get_experiment_directory(experiment_directory)
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
        self.features = [
            feature for feature in self.features if feature not in self.exclude_features
        ]
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
        return MODELS_DIRECTORY / self.folder_name / FEATURES_SELECTION_DIRACTORY

    @property
    def eval_configuration(self):
        """Return evaluation configuration."""
        return self.configuration["evaluation"]

    @property
    def validation_strategy(self):
        """Return validation_strategy."""
        return self.configuration["processing"].get("validation_strategy", True)

    @property
    def exclude_features(self):
        """Return the excluded features list."""
        return [
            feature.lower()
            for feature in self.configuration["processing"].get("exclude_features", [])
        ]

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
    def eval_exp(self, comparison_score_metrics):
        """Evaluation method."""

    @abstractmethod
    def inference(self, data: pd.DataFrame, save_df: bool, file_name: str = ""):
        """Inference method."""

    @abstractmethod
    def plot_comparison_score_vs_predictions(
        self, comparison_score_metrics, predictions_metrics: pd.DataFrame
    ):
        """Process metrics data and plot scores of the comparison score and the predications."""

    def initialize_checkpoint_directory(self, tuning_option: bool = False):
        """Init a ckpt directory."""
        self.experiment_directory.mkdir(exist_ok=True, parents=True)
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
        if not tuning_option:
            self.experiment_directory.mkdir(exist_ok=True, parents=True)
            self.checkpoint_directory.mkdir(exist_ok=True, parents=True)
            self.prediction_directory.mkdir(exist_ok=True, parents=True)
            self.eval_directory.mkdir(exist_ok=True, parents=True)
            self.curve_plot_directory.mkdir(exist_ok=True, parents=True)
            save_features(self.features, self.experiment_directory)

    # Not needed for the moment
    def load_data_set(self) -> None:
        """Load data set."""
        if self.train_data_path:
            self.train_data = Datasetold(
                data_path=self.train_data_path,
                features=self.features,
                target=self.label_name,
                configuration=self.configuration["processing"],
                is_train=True,
                is_unlabeled=False,
                experiment_path=MODELS_DIRECTORY / self.folder_name,
            ).process_data()

        if self.unlabeled_path:
            self.unlabeled_data = Datasetold(
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

        self.test_data = Datasetold(
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
            self.features = load_features(self.experiment_directory / "features")
        else:
            sys.exit(
                (
                    f"Experiment with name {self.experiment_name} is not available."
                    " Please train the model first or change the save_model argument "
                    "in the model config file."
                )
            )

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
        model.eval_model(data=self.test_data(), data_name="test", evaluator=self.evaluator)

        return model

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

    def eval_test(self) -> Dict[str, Any]:
        """Eval method for single data set."""
        test_data = self.inference(
            self.test_data(), file_name=self.test_data_path.stem, save_df=True
        )
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

    def columns_to_save(self, is_train: bool = True) -> List[str]:
        """Return list of columns to be saved."""
        columns = self.test_data.ids_columns + self.features
        columns.append(self.label_name)

        if is_train:
            return columns + ["prediction"]

        columns = columns + self.prediction_columns_name
        if hasattr(self, "kfold_prediction_name"):
            columns = columns + self.kfold_prediction_name
        return columns

    def plotting_summary_scores(self, results: pd.DataFrame):
        """Plot the computed scores for Ig model and the comparasion score."""
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        palette = {
            "IG_model": np.asarray([38, 122, 55]) / 255,
            "Comparison score": np.asarray([42, 203, 74]) / 255,
        }
        for i, split in enumerate(["train", "validation", "test"]):
            split_results = results[results.split == split]
            sns.barplot(
                data=split_results,
                x="prediction",
                y="topk",
                hue="type",
                ax=axs[0, i],
                palette=palette,
            )
            sns.barplot(
                data=split_results,
                x="prediction",
                y="roc",
                hue="type",
                ax=axs[1, i],
                palette=palette,
            )
            axs[0, i].legend(loc="lower left")
            axs[1, i].legend(loc="lower left")
            axs[0, i].set_title(f"Global TopK ({split})")
            axs[1, i].set_title(f"AUC-ROC  ({split})")
            axs[0, i].set_xticklabels(axs[0, i].get_xticklabels(), rotation=45)
            axs[1, i].set_xticklabels(axs[1, i].get_xticklabels(), rotation=45)

            fig.tight_layout()
            fig.savefig(self.eval_directory / "ScoreSummary.png")
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
