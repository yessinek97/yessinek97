"""File that contains the RandomForestModel model that will be used in the experiment."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ig.models.base_model import BaseModel, original_stdout
from ig.utils.io import save_as_pkl


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
