"""Class used to fine tuning the model parameters."""
import logging
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, tpe
from hyperopt.fmin import fmin

import ig.models as src_model
from ig import (
    FEATURES_SELECTION_DIRECTORY,
    KFOLD_MODEL_NAME,
    KFOLD_MULTISEED_MODEL_NAME,
    MODELS_DIRECTORY,
    SINGLE_MODEL_NAME,
)
from ig.constants import TuningOptParamsType
from ig.cross_validation.base import create_model
from ig.dataset.dataset import Dataset
from ig.models import BaseModelType
from ig.utils.cross_validation import (
    convert_int_params,
    get_model_by_name,
    load_features,
    save_features,
)
from ig.utils.general import generate_random_seeds
from ig.utils.io import save_as_pkl, save_yml
from ig.utils.metrics import topk_global

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
    ) -> None:
        """Initialize the Tuning experiment."""
        self.train_data = train_data
        self.test_data = test_data
        self.unlabeled_path = unlabeled_path
        self.configuration = configuration
        self.experiment_name = experiment_name
        self.experiment_param = experiment_param
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

        if sub_folder_name:
            self.experiment_directory = (
                MODELS_DIRECTORY
                / folder_name
                / self.model_type
                / self.experiment_name
                / str(sub_folder_name)
            )
        else:
            ValueError("sub_folder_name should be defined.")

        self.features_directory = MODELS_DIRECTORY / self.folder_name / FEATURES_SELECTION_DIRECTORY
        self.features_file_path = (
            features_file_path
            if features_file_path
            else self.features_directory / self.configuration["features"]
        )

        self.features = [x.lower() for x in load_features(self.features_file_path)]
        self.model_configuration = deepcopy(self.configuration["model"])
        self.initialize_experiment_directory()

        self.int_columns: List[str] = []

        self.save_model = False
        self.model_seed: bool
        self.split_seed: bool
        self.list_of_seeds: List[int]

    @property
    def nbr_trials(self) -> int:
        """Return nbr of trials attribute."""
        return self.configuration["tuning"]["nbr_trials"]

    @property
    def maximize(self) -> bool:
        """Return maximize attribute."""
        return self.configuration["tuning"]["maximize"]

    def initialize_experiment_directory(self) -> None:
        """Initialize the experiment directory."""
        self.experiment_directory.mkdir(exist_ok=True, parents=True)
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
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

    def single_fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> BaseModelType:
        """This method is used to fit and evaluate models using for single or kfold experiments."""
        model = create_model(
            model_cls=self.model_cls,
            model_path=None,
            features=self.features,
            model_params=self.configuration["model"]["model_params"],
            model_general_config=self.configuration["model"]["general_params"],
            label_name=self.configuration["label"],
            folder_name=self.folder_name,
            experiment_name=self.experiment_name,
            save_model=self.save_model,
            prediction_name="prediction",
            dataset_name=self.test_data.dataset_name,
        )
        model.fit(train, validation)

        return model

    def single_validation(self, model_configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparams for single model experiment."""
        model_configuration["model_params"] = convert_int_params(
            self.int_columns, model_configuration["model_params"]
        )
        self.configuration["model"] = model_configuration
        model = self.single_fit(self.train_split, self.validation_split)
        pred = model.predict(self.test_data(), with_label=False)
        score = topk_global(self.test_data("label"), pred)[0]
        if self.maximize:
            score *= -1
        return {"loss": score, "status": STATUS_OK, "params": model_configuration}

    def tune_single_validation(
        self, model_configuration_spaces: Dict[str, Any]
    ) -> TuningOptParamsType:
        """Optimize hyperparams for single model experiment."""
        return self.optimize_parms(self.single_validation, model_configuration_spaces)

    def kfold_fit(
        self,
        train: pd.DataFrame,
        split_column: str,
    ) -> Dict[str, BaseModelType]:
        """Train models using k-fold cross-validation on the given dataset.

        Args:
            train (pd.DataFrame): The training dataset.
            split_column (str): The column in the train dataset that indicates the split.

        Returns:
            Dict[str, BaseModelType]: A dictionary mapping split values to the trained models.
        """
        models = {}
        for split in np.sort(train[split_column].unique()):
            train_data = train[train[split_column] != split]
            validation_data = train[train[split_column] == split]
            model = self.single_fit(
                train_data,
                validation_data,
            )
            models[split] = model
        return models

    def kfold_validation(self, model_configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Perform k-fold cross-validation using the provided model_configuration.

        Args:
            self: The object instance.
            model_configuration (Dict[str, Any]): A dictionary containing model parameters.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - "loss": The loss score of the model on the test data.
                - "status": The status of the validation process.
                - "params": The updated model configuration.
        """
        model_configuration["model_params"] = convert_int_params(
            self.int_columns, model_configuration["model_params"]
        )
        self.configuration["model"] = model_configuration
        models = self.kfold_fit(self.train_data(), split_column=self.split_column)

        preds = []
        for _, i_model in models.items():
            preds.append(i_model.predict(self.test_data(), with_label=False))

        preds = np.mean(preds, axis=0)
        # TODO change topk_global to configurable param
        score = topk_global(self.test_data("label"), preds)[0]
        if self.maximize:
            score *= -1
        return {"loss": score, "status": STATUS_OK, "params": model_configuration}

    def tune_kfold_validation(
        self, model_configuration_spaces: Dict[str, Any]
    ) -> TuningOptParamsType:
        """Optimize hyperparameters for K-fold experiment."""
        return self.optimize_parms(self.kfold_validation, model_configuration_spaces)

    def multi_seed_kfold_validation(self, model_configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-seed k-fold cross-validation using the provided model_configuration.

        Args:
            model_configuration (Dict[str, Any]): A dictionary containing model parameters.
            list_seeds (List[str]): A list of seeds

        Returns:
            Dict[str, Any]: A dictionary
        """
        model_configuration["model_params"] = convert_int_params(
            self.int_columns, model_configuration["model_params"]
        )
        self.configuration["model"] = model_configuration

        models_seeds: Dict[int, Dict[str, BaseModelType]] = {}
        for seed in self.list_of_seeds:
            if self.split_seed:
                self.train_data.kfold_split(self.split_column, seed)
            if self.model_seed:
                model_params = self.model_configuration["model_params"]
                model_params["seed"] = seed
                self.model_configuration["model_params"] = model_params

            models = self.kfold_fit(self.train_data(), split_column=self.split_column)
            models_seeds[seed] = models

        preds = []
        for seed in models_seeds:
            for _, i_model in models_seeds[seed].items():
                preds.append(i_model.predict(self.test_data(), with_label=False))

        preds = np.mean(preds, axis=0)
        # TODO change topk_global to configurable param
        score = topk_global(self.test_data("label"), preds)[0]
        if self.maximize:
            score *= -1
        return {"loss": score, "status": STATUS_OK, "params": model_configuration}

    def tune_multi_seed_kfold_validation(
        self, model_configuration_spaces: Dict[str, Any]
    ) -> TuningOptParamsType:
        """Optimize hyperparameters for multi-seed K-fold experiment."""
        return self.optimize_parms(self.multi_seed_kfold_validation, model_configuration_spaces)

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

    def train(self) -> Dict[str, Union[str, float]]:
        """Train method."""
        if self.experiment_name == SINGLE_MODEL_NAME:
            self._initialize_single_fit()
            tune_fn = self.tune_single_validation

        elif self.experiment_name == KFOLD_MODEL_NAME:
            self._initialize_multi_fit()
            tune_fn = self.tune_kfold_validation
        elif self.experiment_name == KFOLD_MULTISEED_MODEL_NAME:
            self._initialize_multi_seed_fit()
            tune_fn = self.tune_multi_seed_kfold_validation
        else:
            raise NotImplementedError(
                "{experiment_name} is not defined; "
                "choose from: [SingleModel, KfoldExperiment,"
                " KfoldMultiSeedExperiment]"
            )
        self.model_configuration = self.convert_model_params_to_hp(self.model_configuration)
        results, all_trials = tune_fn(model_configuration_spaces=self.model_configuration)
        save_yml(
            results, self.experiment_directory / f"{self.experiment_name}_best_model_params.yml"
        )
        save_as_pkl(all_trials, self.experiment_directory / f"{self.experiment_name}_trials.pkl")
        score = -1 * results["loss"] if self.maximize else results["loss"]
        return {
            "model": self.model_type,
            "experiment": self.experiment_name,
            "features": self.features_list_path,
            "score": score,
        }

    def _initialize_single_fit(self) -> None:
        """Initialize data for single model."""
        self.validation_column = self.experiment_param["validation_column"]
        if isinstance(self.train_data, Dataset):

            if self.validation_column not in self.train_data().columns:
                raise KeyError(f"{self.validation_column} column is missing")
            self.validation_split = self.train_data()[
                self.train_data()[self.validation_column] == 1
            ]
            self.train_split = self.train_data()[self.train_data()[self.validation_column] == 0]

    def _initialize_multi_fit(self) -> None:
        """Initialize data for Kfold model."""
        self.split_column = self.experiment_param["split_column"]
        if isinstance(self.train_data, Dataset):
            if self.split_column not in self.train_data().columns:
                raise KeyError(f"{self.split_column} column is missing")

    def _initialize_multi_seed_fit(self) -> None:
        """Initialize data for multi-seed Kfold model."""
        self.split_column = self.experiment_param["split_column"]
        self.list_of_seeds = self.experiment_param.get("seeds", [])
        self.nbr_seeds = self.experiment_param.get("nbr_seeds", None)

        if len(self.list_of_seeds) == 0 and self.nbr_seeds is None:
            raise ValueError(
                "Please set seeds or nbr_seeds in experiment param in the configuration file."
            )

        if len(self.list_of_seeds) > 0 and self.nbr_seeds is not None:
            raise ValueError(
                "Please set only seeds or nbr_seeds in experiment param in the configuration file."
            )

        if len(self.list_of_seeds) == 0 and self.nbr_seeds is not None:
            logger.info("Generating %d random seeds", self.nbr_seeds)
            self.list_of_seeds = generate_random_seeds(self.nbr_seeds)

        self.nbr_seeds = len(self.list_of_seeds)
        self.split_seed = self.experiment_param.get("split_seed", False)
        self.model_seed = self.experiment_param.get("model_seed", False)
        if not (self.split_seed or self.model_seed):
            raise ValueError("Please set split_seed or model_seed to True in experiment param.")

        # Update saved configuration file
        self.experiment_param["seeds"] = self.list_of_seeds
        self.experiment_param["nbr_seeds"] = self.nbr_seeds
        self.configuration["experiments"][self.experiment_name] = self.experiment_param
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")

        self.train_data.force_validation_strategy()
