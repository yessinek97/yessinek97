"""File that contains the LogisticRegressionModel model that will be used in the experiment."""

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ig.models.base_model import BaseModel, original_stdout
from ig.src.utils import save_as_pkl


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
