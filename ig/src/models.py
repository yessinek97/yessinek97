"""File that contains all the model that will be used in the experiment."""
from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC

from ig.src.evaluation import Evaluation
from ig.src.logger import ModelLogWriter, get_logger
from ig.src.utils import save_as_pkl

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

    def eval_model(self, data: pd.DataFrame, data_name: str, evaluator: Evaluation) -> None:
        """Eval method."""
        data = data.copy()
        data[self.prediction_name] = self.predict(data, with_label=False)
        evaluator.compute_metrics(
            data=data, prediction_name=self.prediction_name, data_name=data_name
        )

    def generate_shap_values(self, data: pd.DataFrame) -> Any:
        """Generate shap values."""
        return shap.TreeExplainer(self.model).shap_values(data)

    @abstractmethod
    def _create_matrix(self, data: pd.DataFrame, with_label: bool) -> Any | None:
        """Return the correct data structure. Object that is required by the model."""


warnings.filterwarnings("ignore")


class XgboostModel(BaseModel):
    """Xgboost model class."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        dtrain = self._create_matrix(train_data)
        dval = self._create_matrix(val_data)

        self.model = xgb.train(
            params=self.parameters,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dval, "valid")],
            num_boost_round=self.other_params.get("num_boost_round", 20000),
            verbose_eval=self.other_params.get("verbose_eval", True),
            early_stopping_rounds=self.other_params.get("early_stopping_rounds", 20),
        )

        self.shap_values = self.generate_shap_values(train_data[self.features])

        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        dmatrix = self._create_matrix(data, with_label)
        best_iteration = self.model.best_iteration
        return self.model.predict(dmatrix, ntree_limit=best_iteration)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any | None:
        """Data model creation."""
        label = data[self.label_name] if with_label else None
        return xgb.DMatrix(data=data[self.features], label=label, feature_names=self.features)


class LgbmModel(BaseModel):
    """This is an implementation of lgbm model.Based on the Native Microsoft Implementation."""

    def generate_shap_values(self, data: pd.DataFrame) -> Any:
        """Generate shap values."""
        return shap.TreeExplainer(self.model).shap_values(data)[1]

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        # Load data
        true_data = train_data
        train_data = self._create_matrix(train_data)
        val_data = self._create_matrix(val_data, with_label=True)

        # train model
        self.model = lgb.train(
            self.parameters,
            train_data,
            num_boost_round=self.other_params.get("num_boost_round"),
            valid_sets=[val_data],
            verbose_eval=self.other_params["verbose_eval"],
            early_stopping_rounds=self.other_params.get("early_stopping_rounds"),
        )
        self.shap_values = self.generate_shap_values(true_data[self.features])
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        return self.model.predict(data[self.features], num_iteration=self.model.best_iteration)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any | None:
        """Data model creation."""
        if with_label:
            return lgb.Dataset(
                data[self.features], label=data[self.label_name], feature_name=self.features
            )

        return lgb.Dataset(data[self.features], feature_name=self.features)


class CatBoostModel(BaseModel):
    """This is an implementation of catboost model.Based on the Native Implementation."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        # create model
        model = CatBoostClassifier(**self.parameters)
        # train model
        self.model = model.fit(
            X=train_data[self.features].values,
            y=train_data[self.label_name].values,
            eval_set=(val_data[self.features].values, val_data[self.label_name].values),
            log_cout=sys.stdout,
            log_cerr=sys.stderr,
        )
        self.shap_values = self.generate_shap_values(train_data[self.features])
        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        return self.model.predict_proba(data[self.features].values)[:, 1]

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any | None:
        """Data model creation."""


class LabelPropagationModel(BaseModel):
    """Label Propagation model class."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        dtrain = self._create_matrix(train_data)
        labels = self._create_matrix(train_data, with_label=False)
        self.model = LabelPropagation(**self.parameters)
        self.model = self.model.fit(dtrain, labels)
        self.shap_values = None
        log.info(" Shap values for this model type not supported")

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        dmatrix = self._create_matrix(data, with_label)
        return self.model.predict(dmatrix)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Data model creation."""
        if with_label:
            return data[self.features].to_numpy()
        return data[self.label_name].to_numpy()


class LogisticRegressionModel(BaseModel):
    """This is an implementation of LogisticRegression model.

    Based on the Native Implementation.
    """

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:  # pylint disable=W0613
        """Fitting model."""
        self.model = LogisticRegression().fit(
            train_data[self.features], train_data[self.label_name]
        )

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        return self.model.predict_proba(data[self.features].values)[:, 1]

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Data model creation."""


class RandomForestModel(BaseModel):
    """This is an implementation of Random Forest model.

    Based on the Native Implementation.
    """

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:  # pylint disable=W0613
        """Fitting model."""
        self.model = RandomForestClassifier(**self.parameters).fit(
            train_data[self.features], train_data[self.label_name]
        )

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data: pd.DataFrame, with_label: bool = True) -> np.ndarray:
        """Prediction method."""
        return self.model.predict(data[self.features])

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Data model creation."""


class SupportVectorMachineModel(BaseModel):
    """This is an implementation of support vector machine model."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        # create model
        model = SVC(**self.parameters)
        # train model
        self.model = model.fit(
            X=train_data[self.features].values, y=train_data[self.label_name].values
        )
        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Prediction method."""
        return self.model.predict(data[self.features].values)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Data model creation."""


BaseModelType = Union[
    XgboostModel,
    LgbmModel,
    CatBoostClassifier,
    LabelPropagationModel,
    LogisticRegressionModel,
    RandomForestModel,
    SupportVectorMachineModel,
]

TrainSingleModelType = BaseModelType
TrainKfoldType = Dict[str, BaseModelType]
TrainDoubleKfold = Dict[str, BaseModelType]
TrainMultiSeedKfold = Dict[str, Dict[str, BaseModelType]]
TrainTuneType = Dict[str, Any]
TrainType = Union[
    TrainSingleModelType, TrainKfoldType, TrainDoubleKfold, TrainMultiSeedKfold, TrainTuneType
]
