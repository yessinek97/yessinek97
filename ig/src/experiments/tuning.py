"""Class used to fine tuning the model parameters."""
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, tpe
from hyperopt.fmin import fmin

import ig.src.models as src_model
from ig import FEATURES_SELECTION_DIRECTORY, MODELS_DIRECTORY
from ig.constants import TuningOptParamsType
from ig.dataset.dataset import Dataset
from ig.src.experiments.base import create_model
from ig.src.metrics import topk_global
from ig.src.models import BaseModelType, TrainTuneType
from ig.src.utils import (
    convert_int_params,
    get_model_by_name,
    load_features,
    save_as_pkl,
    save_features,
    save_yml,
)

logger: Logger = logging.getLogger("Tuning")


class Tuning:
    """Tune model hyperparams Class."""

    def __init__(
        self,
        train_data: Dataset,
        test_data: Dataset,
        configuration: Dict[str, Any],
        folder_name: str,
        experiment_param: Dict[str, Any],
        is_compute_metrics: bool,
        experiment_name: str,
        features_list_path: str,
        features_configuration_path: Path,
        sub_folder_name: Optional[str] = None,
        unlabeled_path: Optional[str] = None,
        features_file_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the Tuning experiment."""
        self.train_data = train_data
        self.test_data = test_data
        self.unlabeled_path = unlabeled_path
        self.configuration = configuration
        self.experiment_name = experiment_name
        self.folder_name = folder_name
        self.sub_folder_name = sub_folder_name
        self.features_configuration_path = features_configuration_path
        self.features_file_path = features_file_path
        self.features_list_path = features_list_path
        self.is_compute_metrics = is_compute_metrics
        self.model_type = self.configuration["model_type"]
        self.model_cls = get_model_by_name(src_model, self.model_type)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        self.experiment_directory = MODELS_DIRECTORY / folder_name / self.model_type
        self.plot_shap_values = bool(kwargs.get("plot_shap_values", False))
        self.plot_kfold_shap_values = bool(kwargs.get("plot_kfold_shap_values", False))
        self.features_directory = MODELS_DIRECTORY / self.folder_name / FEATURES_SELECTION_DIRECTORY
        self.features_file_path = (
            features_file_path
            if features_file_path
            else self.features_directory / self.configuration["features"]
        )
        self.checkpoint_directory = self.experiment_directory / "checkpoint"

        self.features = [x.lower() for x in load_features(self.features_file_path)]
        self.model_configuration = self.configuration["model"]
        self.initialize_checkpoint_directory()
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
        if experiment_name == "SingleModel":
            self._initialize_single_fit(experiment_param=experiment_param)
        if experiment_name == "KfoldExperiment":
            self._initialize_multi_fit(experiment_param=experiment_param)
        self.int_columns: List[str] = []

        self.model_configuration = self.convert_model_params_to_hp(self.model_configuration)
        self.save_model = False

    @property
    def nbr_trials(self) -> int:
        """Return nbr of trials attribute."""
        return self.configuration["tuning"]["nbr_trials"]

    @property
    def maximize(self) -> bool:
        """Return maximize attribute."""
        return self.configuration["tuning"]["maximize"]

    def initialize_checkpoint_directory(self) -> None:
        """Init a ckpt directory."""
        self.experiment_directory.mkdir(exist_ok=True, parents=True)
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
        self.experiment_directory.mkdir(exist_ok=True, parents=True)
        self.checkpoint_directory.mkdir(exist_ok=True, parents=True)

        save_features(self.features, self.experiment_directory)

    def optimize_parms(
        self, train_func: Callable, model_param_spaces: Dict[str, Any]
    ) -> TuningOptParamsType:
        """Optimize the model hyperparams."""
        trials = Trials()
        fmin(
            fn=train_func,
            space=model_param_spaces,
            algo=tpe.suggest,
            max_evals=self.nbr_trials,
            trials=trials,
            verbose=1,
        )
        best_index = np.argmin(trials.losses())
        results = trials.trials[best_index]["result"]
        return results, trials.trials

    def _train_function(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        split: Optional[int] = None,
        prediction_name: str = "prediction",
        sub_model_directory: Optional[str] = None,
        kfold: bool = False,
    ) -> BaseModelType:
        """This method is used to fit and evaluate models using for single or kfold experiments."""
        if kfold and split:

            self.checkpoint_path = (
                self.checkpoint_directory / sub_model_directory
                if sub_model_directory
                else self.checkpoint_directory
            )

            self.model_name = f"split_{split}"
        self.model_path = (
            (
                self.checkpoint_path / self.model_name / "model.pkl"
                if self.model_name
                else self.checkpoint_path / "model.pkl"
            )
            if self.checkpoint_path
            else None
        )
        model = create_model(
            model_cls=self.model_cls,
            model_path=self.model_path,
            features=self.features,
            model_params=self.configuration["model"]["model_params"],
            model_general_config=self.configuration["model"]["general_params"],
            label_name=self.configuration["label"],
            folder_name=self.folder_name,
            experiment_name=self.experiment_name,
            save_model=self.save_model,
            prediction_name=prediction_name,
            dataset_name=self.test_data.dataset_name,
        )
        model.fit(train, validation)

        return model

    def tune_single_model(self, model_configuration_spaces: Dict[str, Any]) -> TuningOptParamsType:
        """Optimize hyperparams for single model experiment."""

        def single_fit(model_configuration: Dict[str, Any]) -> Dict[str, Any]:
            """Optimize hyperparams for single model experiment."""
            model_configuration["model_params"] = convert_int_params(
                self.int_columns, model_configuration["model_params"]
            )
            self.configuration["model"] = model_configuration
            model = self._train_function(self.train_split, self.validation_split)
            pred = model.predict(self.test_data(), with_label=False)
            score = topk_global(self.test_data("label"), pred)[0]
            if self.maximize:
                score *= -1
            return {"loss": score, "status": STATUS_OK, "params": model_configuration}

        return self.optimize_parms(single_fit, model_configuration_spaces)

    def fit_kfold_models(
        self,
        train: pd.DataFrame,
        split_column: str,
        sub_model_directory: Optional[str] = None,
        multi_seed: bool = False,
    ) -> Dict[str, BaseModelType]:
        """This method is used to fit the kfolds experiments."""
        models = {}
        display = "##### Start Split train #####"
        for split in np.sort(train[split_column].unique()):

            logger.info("           -split %s", split)
            prediction_name = (
                f"prediction_{sub_model_directory}_{split}" if multi_seed else f"prediction_{split}"
            )
            display += f"#### split {split+1} #### "
            train_data = train[train[split_column] != split]
            validation_data = train[train[split_column] == split]
            model = self._train_function(
                train_data,
                validation_data,
                split,
                prediction_name,
                sub_model_directory,
                kfold=True,
            )
            display += "##### End Split train #####"
            models[split] = model

        logger.debug(display)

        return models

    def tune_kfold_model(self, model_configuration_spaces: Dict[str, Any]) -> TuningOptParamsType:
        """Optimize hyperparams for Kfold experiment."""

        def multiple_fit(model_configuration: Dict[str, Any]) -> Dict[str, Any]:
            """Optimize hyperparams for Kfold experiment."""
            model_configuration["model_params"] = convert_int_params(
                self.int_columns, model_configuration["model_params"]
            )
            self.configuration["model"] = model_configuration
            models = self.fit_kfold_models(
                self.train_data(), split_column=self.split_column, sub_model_directory=None
            )

            preds = []
            for _, i_model in models.items():
                preds.append(i_model.predict(self.test_data(), with_label=False))

            preds = np.mean(preds, axis=0)
            # TODO change topk_global to configurable param
            score = topk_global(self.test_data("label"), preds)[0]
            if self.maximize:
                score *= -1
            return {"loss": score, "status": STATUS_OK, "params": model_configuration}

        return self.optimize_parms(multiple_fit, model_configuration_spaces)

    def convert_model_params_to_hp(self, model_param: Dict[str, Any]) -> Dict[str, Any]:
        """Convert model params to hp object."""
        model_configuration_dynamic = {}
        for key, value in zip(
            model_param["model_params"].keys(), model_param["model_params"].values()
        ):
            if isinstance(value, dict):
                model_configuration_dynamic[key] = hp.quniform(
                    key, value["low"], value["high"], value["q"]
                )
                if value["type"] == "int":
                    self.int_columns.append(key)
            else:
                model_configuration_dynamic[key] = value
        model_param["model_params"] = model_configuration_dynamic
        return model_param

    def train(self) -> TrainTuneType:  # pylint: disable=W0221
        """Train method."""
        if self.experiment_name == "SingleModel":
            results, all_trials = self.tune_single_model(
                model_configuration_spaces=self.model_configuration
            )

        elif self.experiment_name == "KfoldExperiment":
            results, all_trials = self.tune_kfold_model(
                model_configuration_spaces=self.model_configuration
            )
        else:
            raise NotImplementedError(
                "{experiment_name} is not defined; "
                "choose from: [SingleModel, KfoldExperiment,"
                "SingKfoldModel]"
            )

        path = self.experiment_directory / str(self.sub_folder_name)
        path.mkdir(exist_ok=True, parents=True)
        save_yml(results, path / f"{self.experiment_name}_{self.model_type}_best_model_params.yml")
        save_as_pkl(all_trials, path / f"{self.experiment_name}_{self.model_type}_trials.pkl")
        score = -1 * results["loss"] if self.maximize else results["loss"]
        return {
            "model": self.model_type,
            "experiment": self.experiment_name,
            "features": self.features_list_path,
            "score": score,
        }

    def _initialize_single_fit(self, experiment_param: Dict[str, Any]) -> None:
        """Initialize data for single model."""
        self.validation_column = experiment_param["validation_column"]
        if isinstance(self.train_data, Dataset):

            if self.validation_column not in self.train_data().columns:
                raise KeyError(f"{self.validation_column} column is missing")
            self.validation_split = self.train_data()[
                self.train_data()[self.validation_column] == 1
            ]
            self.train_split = self.train_data()[self.train_data()[self.validation_column] == 0]

    def _initialize_multi_fit(self, experiment_param: Dict[str, Any]) -> None:
        """Initialize data for Kfold model."""
        self.split_column = experiment_param["split_column"]
        if isinstance(self.train_data, Dataset):
            if self.split_column not in self.train_data().columns:
                raise KeyError(f"{self.split_column} column is missing")
