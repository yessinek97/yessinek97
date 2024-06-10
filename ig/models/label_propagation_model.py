"""File that contains the Label Propagation model that will be used in the experiment."""
from typing import Any

import pandas as pd
from sklearn.semi_supervised import LabelPropagation

from ig.models.base_model import BaseModel, log, original_stdout
from ig.utils.io import save_as_pkl


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
