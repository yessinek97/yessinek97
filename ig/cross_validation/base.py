"""File that contains the implementation of the base classes."""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ig.models as src_model
from ig import FEATURES_SELECTION_DIRECTORY, MODELS_DIRECTORY, SINGLE_MODEL_NAME
from ig.constants import (
    EvalExpType,
    EvalExpTypeBase,
    ExpPredictTypeBase,
    InferenceExpTypeBase,
    MetricsEvalType,
)
from ig.dataset.dataset import Dataset
from ig.models import BaseModelType
from ig.utils.cross_validation import (
    get_model_by_name,
    load_features,
    plotting_kfold_shap_values,
    plotting_shap_values,
    save_features,
)
from ig.utils.evaluation import Evaluation
from ig.utils.io import load_yml, save_yml
from ig.utils.logger import get_logger
from ig.utils.torch import empty_cache

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
        experiment_name: str,
        features_configuration_path: Path,
        sub_folder_name: Optional[str] = None,
        unlabeled_path: Optional[str] = None,
        experiment_directory: Optional[Path] = None,
        features_file_path: Optional[str] = None,
        is_compute_metrics: bool = True,
    ) -> None:
        """Class init."""
        self.train_data = train_data
        self.test_data = test_data
        self.unlabeled_path = unlabeled_path
        self.test_data_path = test_data.data_path if isinstance(test_data, Dataset) else None
        self.train_data_path = train_data.data_path if isinstance(train_data, Dataset) else None
        self.dataset_name = self.test_data.dataset_name
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

        self.save_model = True
        self.evaluator = Evaluation(
            label_name=self.label_name,
            eval_configuration=self.eval_configuration,
            curve_plot_directory=self.curve_plot_directory,
            is_compute_metrics=is_compute_metrics,
        )
        self.features_configuration = load_yml(features_configuration_path)

        self.plot_shap_values: bool
        self.plot_kfold_shap_values: Optional[bool]
        self.prediction_columns_name: List[str] = []
        self.statistics_opts: List[str]
        self.kfold_prediction_name: List[str]

    @property
    def model_type(self) -> str:
        """Type of the model."""
        return self.check_and_return_model_type(self.configuration["model_type"])

    @property
    def label_name(self) -> str:
        """Label name."""
        return self.configuration["label"]

    @property
    def is_checkpoint_directory_empty(self) -> bool:
        """Checking if directory is empty."""
        if self.checkpoint_directory.exists():
            return not any(self.checkpoint_directory.iterdir())
        return False

    @property
    def task(self) -> Optional[str]:
        """Return the task of the run."""
        if "8" in self.label_name:
            return "CD8"
        if "4" in self.label_name:
            return "CD4"
        return None

    @property
    def features_directory(self) -> Path:
        """Return features list directory."""
        return MODELS_DIRECTORY / self.folder_name / FEATURES_SELECTION_DIRECTORY

    @property
    def eval_configuration(self) -> Dict[str, Any]:
        """Return evaluation configuration."""
        return self.configuration["evaluation"]

    @property
    def validation_strategy(self) -> bool:
        """Return validation_strategy."""
        return self.configuration["processing"].get("validation_strategy", True)

    @property
    def exclude_features(self) -> List[str]:
        """Return the excluded features list."""
        return [
            feature.lower()
            for feature in self.configuration["processing"].get("exclude_features", [])
        ]

    @property
    def model_configuration(self) -> Dict[str, Any]:
        """Model cfg."""
        return self.configuration["model"]

    @model_configuration.setter
    def model_configuration(self, model_configuration: Dict[str, Any]) -> Any:
        self.configuration["model"] = model_configuration

    @property
    def ids_columns(self) -> List[str]:
        """Return ids columns list."""
        ids = [id_.lower() for id_ in self.features_configuration.get("ids", [])]
        if self.id_column not in ids:
            ids.append(self.id_column)
        return ids

    @property
    def id_column(self) -> str:
        """Return main id column list."""
        return self.features_configuration["id"].lower()

    @abstractmethod
    def train(self) -> None:
        """Training method."""

    @abstractmethod
    def predict(self, save_df: bool) -> ExpPredictTypeBase:
        """Predict method."""

    @abstractmethod
    def eval_exp(self, comparison_score_metrics: Optional[pd.DataFrame]) -> EvalExpTypeBase:
        """Evaluation method."""

    @abstractmethod
    def inference(
        self, data: pd.DataFrame, save_df: bool, file_name: str = ""
    ) -> InferenceExpTypeBase:
        """Inference method."""

    @abstractmethod
    def plot_comparison_score_vs_predictions(
        self, comparison_score_metrics: Optional[pd.DataFrame], predictions_metrics: pd.DataFrame
    ) -> None:
        """Process metrics data and plot scores of the comparison score and the predictions."""

    def restore(self) -> None:
        """Restoring a ckpt."""
        if self.experiment_directory.exists() & (not self.is_checkpoint_directory_empty):
            self.features = load_features(self.experiment_directory / "features")
        else:
            sys.exit(
                f"Experiment with name {self.experiment_name} is not available."
                " Please train the model first or change the save_model argument "
                "in the model config file."
            )

    def single_fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        checkpoints: Optional[Path] = None,
        model_name: Optional[str] = None,
        prediction_name: str = "prediction",
    ) -> BaseModelType:
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
        model = create_model(
            model_cls=self.model_cls,
            model_path=model_path,
            features=self.features,
            model_params=self.model_configuration["model_params"],
            model_general_config=self.model_configuration["general_params"],
            label_name=self.label_name,
            folder_name=self.folder_name,
            experiment_name=self.experiment_name,
            save_model=self.save_model,
            prediction_name=prediction_name,
            dataset_name=self.dataset_name,
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

        model.eval_model(data=train, split_name="train", evaluator=self.evaluator)
        model.eval_model(data=validation, split_name="validation", evaluator=self.evaluator)
        model.eval_model(data=self.test_data(), split_name="test", evaluator=self.evaluator)

        return model

    def multiple_fit(
        self,
        train: pd.DataFrame,
        split_column: str,
        sub_model_directory: Optional[str] = None,
        multi_seed: bool = False,
    ) -> None:
        """Multiple fit model."""
        shap_values = []
        checkpoints = (
            self.checkpoint_directory / sub_model_directory
            if sub_model_directory
            else self.checkpoint_directory
        )

        display = "##### Start Split train #####"
        for split in np.sort(train[split_column].unique()):
            if self.evaluator.print_evals:
                log.info("           -split %s", split)
            prediction_name = (
                f"prediction_{sub_model_directory}_{split}" if multi_seed else f"prediction_{split}"
            )
            display += f"#### split {split+1} #### "
            train_data = train[train[split_column] != split]
            validation_data = train[train[split_column] == split]
            model = self.single_fit(
                train=train_data,
                validation=validation_data,
                checkpoints=checkpoints,
                model_name=f"split_{split}",
                prediction_name=prediction_name,
            )

            display += "##### End Split train #####"
            if self.plot_shap_values:
                shap_values.append(
                    pd.DataFrame(
                        {
                            "features": self.features,
                            "scores": list(np.mean(np.abs(model.shap_values), axis=0)),
                        }
                    )
                )
            del model
            empty_cache()
        if self.plot_shap_values and self.plot_kfold_shap_values:
            shap_values_df = pd.concat(shap_values)
            shap_values_df = (
                shap_values_df.groupby("features").scores.mean().rename("scores").reset_index()
            )
            if multi_seed:
                shap_values_df.to_csv(
                    self.eval_directory / f"shap_values_{sub_model_directory}.csv", index=False
                )
            else:
                plotting_kfold_shap_values(shap_values_df, self.eval_directory / "shap.png")
        if self.evaluator.print_evals:
            log.debug(display)

    def eval_test(self) -> EvalExpType:
        """Eval method for single data set."""
        test_data = self.inference(
            self.test_data(), file_name=self.test_data_path.stem, save_df=True
        )
        if self.experiment_name == SINGLE_MODEL_NAME:
            prediction_columns_name = self.prediction_columns_name
        else:
            prediction_columns_name = self.prediction_columns_name + self.kfold_prediction_name

        for prediction_name in prediction_columns_name:
            log.info("            %s : ", prediction_name)
            self.evaluator.compute_metrics(
                data=test_data,
                prediction_name=prediction_name,
                split_name="test",
                dataset_name=self.dataset_name,
            )
        results = self._parse_metrics_to_data_frame(
            eval_metrics=self.evaluator.get_evals(), file_name=self.test_data_path.stem
        )
        fake_validation = results.copy()
        fake_validation["split"] = "validation"
        results = pd.concat([results, fake_validation])
        if self.experiment_name:
            best_validation_scores, best_test_scores, _ = self.evaluator.get_experiment_best_scores(
                results=results,
                experiment_name=self.experiment_name,
                model_type=self.model_type,
                features_name=self.configuration["features"],
            )
        else:
            raise Exception("No experiment name provided!")

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
            "RandomForestModel",
            "SupportVectorMachineModel",
            "LLMModel",
            "LLMMixedModel",
        ]:
            raise NotImplementedError(
                f"{model_type} is not supported."
                "The possible choices are:"
                "[XgboostModel, LgbmModel, CatBoostModel,"
                " LabelPropagationModel, RandomForestModel, SupportVectorMachineModel"
                " LLMModel, LLMMixedModel]"
            )
        return model_type

    def columns_to_save(
        self, data: pd.DataFrame, is_train: bool = True, multi_seed: bool = False
    ) -> List[str]:
        """Return list of columns to be saved.

        Args:
            data:   dataframe which will be saved
            is_train : if it's  True the function will return (Ids and features) with
            the prediction column name
            multi_seed : if it's True the function will return only Ids and features
        """
        columns = self.ids_columns + self.features

        if self.label_name in data.columns:
            columns.append(self.label_name)

        if is_train:
            return columns + ["prediction"]
        if multi_seed:
            return columns
        columns = columns + self.prediction_columns_name
        if hasattr(self, "kfold_prediction_name"):
            columns = columns + self.kfold_prediction_name
        return columns

    def plotting_summary_scores(self, results: pd.DataFrame) -> None:
        """Plot the computed scores for Ig model and the comparison score."""
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        palette = {
            "IG_model": np.asarray([38, 122, 55]) / 255,
            self.evaluator.comparison_score: np.asarray([42, 203, 74]) / 255,
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
            for container in axs[0, i].containers:
                axs[0, i].bar_label(container, fmt="%0.3f")
            for container in axs[1, i].containers:
                axs[1, i].bar_label(container, fmt="%0.3f")

            axs[0, i].set_title(f"Global TopK {split}")
            axs[1, i].set_title(f"AUC-ROC {split} ")

            axs[0, i].set_xticklabels(axs[0, i].get_xticklabels(), rotation=45)
            axs[1, i].set_xticklabels(axs[1, i].get_xticklabels(), rotation=45)

            fig.tight_layout()
            fig.savefig(self.eval_directory / "ScoreSummary.png")
            plt.close("all")

    @abstractmethod
    def _parse_metrics_to_data_frame(
        self, eval_metrics: MetricsEvalType, file_name: str = "results"
    ) -> pd.DataFrame:
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

    def _save_prediction_name_selector(self, prediction_name_selector: str) -> None:
        """Save prediction name selector."""
        self.configuration["evaluation"]["prediction_name_selector"] = prediction_name_selector
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")

    def initialize_checkpoint_directory(self) -> None:
        """Init a ckpt directory."""
        self.experiment_directory.mkdir(exist_ok=True, parents=True)
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
        self.experiment_directory.mkdir(exist_ok=True, parents=True)
        self.checkpoint_directory.mkdir(exist_ok=True, parents=True)
        self.prediction_directory.mkdir(exist_ok=True, parents=True)
        self.eval_directory.mkdir(exist_ok=True, parents=True)
        self.curve_plot_directory.mkdir(exist_ok=True, parents=True)
        save_features(self.features, self.experiment_directory)


def create_model(
    model_cls: Callable,
    model_path: Optional[Path],
    features: List[str],
    model_params: Dict[str, Any],
    label_name: str,
    model_general_config: Dict[str, Any],
    folder_name: str,
    experiment_name: str,
    prediction_name: str,
    save_model: bool,
    dataset_name: str,
) -> BaseModelType:
    """Creating a model.

    Args:
        model_cls: model class
        model_path: path to the model
        features: list of features
        model_params: model parameters
        label_name:  target name
        model_general_config: general model configuration
        folder_name: main folder name where the run will be saved
        experiment_name: the experiment name (KfoldExperiment,SingleModel,..)
        prediction_name: str  name of the predicted column
        save_model: boolean to control saving the model
        dataset_name: The name of the dataset.

    Return:
        model : an instance of the Models classes
    """
    return model_cls(
        features=features,
        parameters=model_params,
        label_name=label_name,
        prediction_name=prediction_name,
        checkpoints=model_path,
        other_params=model_general_config,
        folder_name=folder_name,
        experiment_name=experiment_name,
        model_type=model_cls.__name__,
        save_model=save_model,
        dataset_name=dataset_name,
    )
