"""File that contains all the model that will be used in the experiment."""
from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import shap

from ig.src.evaluation import Evaluation
from ig.src.logger import ModelLogWriter, get_logger

log = get_logger("Train/model")
original_stdout = sys.__stdout__


class BaseModel(ABC):
    """Basic class Method."""

    def __init__(
        self,
        features: list[str],
        parameters: dict[str, Any],
        label_name: str,
        prediction_name: str,
        checkpoints: Path,
        other_params: dict[str, Any],
        folder_name: str,
        experiment_name: str,
        model_type: str,
        save_model: bool,
        dataset_name: str,
    ) -> None:
        """Init method."""
        self.features = features
        self.parameters = parameters
        self.label_name = label_name
        self.prediction_name = prediction_name
        self.checkpoints = checkpoints
        self.other_params = other_params
        self.folder_name = folder_name
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.shap_values = None
        self.dataset_name = dataset_name
        self.model_logger: ModelLogWriter
        self.model: Any
        self.save_model = save_model
        if self.save_model:
            self.checkpoints.parent.mkdir(parents=True, exist_ok=True)
            self.model_logger = ModelLogWriter(str(self.checkpoints.parent / "model.log"))

    @property
    def model_meta_data(self) -> dict[str, Any]:
        """Model meta data."""
        return {
            "model": self.model,
            "model_params": self.parameters,
            "features": self.features,
            "checkpoints": self.checkpoints,
        }

    @abstractmethod
    def predict(self, data: pd.DataFame, with_label: bool) -> Any:
        """Prediction method."""

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fit method."""

    def eval_model(self, data: pd.DataFrame, split_name: str, evaluator: Evaluation) -> None:
        """Eval method."""
        data = data.copy()

        data[self.prediction_name] = self.predict(data, with_label=False)
        evaluator.compute_metrics(
            data=data,
            prediction_name=self.prediction_name,
            split_name=split_name,
            dataset_name=self.dataset_name,
        )

    def generate_shap_values(self, data: pd.DataFrame) -> Any:
        """Generate shap values."""
        return shap.TreeExplainer(self.model).shap_values(data)

    @abstractmethod
    def _create_matrix(self, data: pd.DataFrame, with_label: bool) -> Any | None:
        """Return the correct data structure. Object that is required by the model."""


warnings.filterwarnings("ignore")
