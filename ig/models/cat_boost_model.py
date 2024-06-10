"""File that contains the CatBoost model that will be used in the experiment."""
import sys
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from ig.models.base_model import BaseModel, original_stdout
from ig.utils.io import save_as_pkl


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

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Data model creation."""
