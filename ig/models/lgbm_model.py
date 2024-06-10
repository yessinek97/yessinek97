"""File that contains the LGBM model that will be used in the experiment."""
from typing import Any

import lightgbm as lgb
import pandas as pd
import shap

from ig.models.base_model import BaseModel, original_stdout
from ig.utils.io import save_as_pkl


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
        verbose_eval = self.other_params["verbose_eval"]
        early_stopping_rounds = self.other_params.get("early_stopping_rounds")

        # train model
        self.model = lgb.train(
            self.parameters,
            train_data,
            num_boost_round=self.other_params.get("num_boost_round"),
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(verbose_eval)],
        )
        self.shap_values = self.generate_shap_values(true_data[self.features])
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        return self.model.predict(data[self.features], num_iteration=self.model.best_iteration)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Data model creation."""
        if with_label:
            return lgb.Dataset(
                data[self.features],
                label=data[self.label_name],
                feature_name=self.features,
            )

        return lgb.Dataset(data[self.features], feature_name=self.features)
