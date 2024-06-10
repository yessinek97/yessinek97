"""File that contains the XGboost model that will be used in the experiment."""

from typing import Any

import pandas as pd
import xgboost as xgb

from ig.models.base_model import BaseModel, original_stdout
from ig.utils.io import save_as_pkl


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
        return self.model.predict(dmatrix, iteration_range=(0, best_iteration + 1))

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Data model creation."""
        label = data[self.label_name] if with_label else None
        return xgb.DMatrix(data=data[self.features], label=label, feature_names=self.features)
