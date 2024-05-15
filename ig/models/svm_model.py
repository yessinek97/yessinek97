"""File that contains the SupportVectorMachineModel model that will be used in the experiment."""
import pandas as pd
from sklearn.svm import SVC

from ig.models.base_model import BaseModel, original_stdout
from ig.src.utils import save_as_pkl


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
