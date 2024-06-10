"""Classes to launch training using a single model."""
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ig import SINGLE_MODEL_NAME
from ig.constants import EvalExpType, ExpPredictType, InferenceExpType, MetricsEvalType
from ig.cross_validation.base import BaseExperiment
from ig.dataset.dataset import Dataset
from ig.utils.io import load_pkl, save_yml
from ig.utils.torch import empty_cache


class SingleModel(BaseExperiment):
    """Class to handle training a single specific model type."""

    def __init__(
        self,
        train_data: Dataset,
        test_data: Dataset,
        configuration: Dict[str, Any],
        folder_name: str,
        experiment_name: str,
        features_configuration_path: Path,
        sub_folder_name: Optional[str] = None,
        unlabeled_path: Optional[str] = None,
        experiment_directory: Optional[Path] = None,
        features_file_path: Optional[str] = None,
        is_compute_metrics: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the single model experiment."""
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            unlabeled_path=unlabeled_path,
            configuration=configuration,
            experiment_name=experiment_name,
            folder_name=folder_name,
            sub_folder_name=sub_folder_name,
            experiment_directory=experiment_directory,
            features_file_path=features_file_path,
            is_compute_metrics=is_compute_metrics,
            features_configuration_path=features_configuration_path,
        )

        self.validation_column = str(kwargs["validation_column"])
        self.plot_shap_values = bool(kwargs.get("plot_shap_values", False))
        if isinstance(self.train_data, Dataset):
            self.initialize_checkpoint_directory()
            if self.validation_column not in self.train_data().columns:
                raise KeyError(f"{self.validation_column} column is missing")
            self.validation_split = self.train_data()[
                self.train_data()[self.validation_column] == 1
            ]
            self.train_split = self.train_data()[self.train_data()[self.validation_column] == 0]
        self.prediction_columns_name = ["prediction"]

    def train(self) -> None:
        """Training method."""
        model = self.single_fit(
            self.train_split,
            self.validation_split,
            self.checkpoint_directory,
            prediction_name=self.prediction_columns_name[0],
        )
        del model
        empty_cache()

    def predict(self, save_df: bool = True) -> ExpPredictType:
        """Predict method."""
        test_data = self.inference(self.test_data(), save_df=save_df, file_name="test")
        validation = self.inference(self.validation_split, save_df=save_df, file_name="validation")
        self.inference(self.train_split, save_df=save_df, file_name="train")
        return validation, test_data

    def inference(
        self, data: pd.DataFrame, save_df: bool = False, file_name: str = ""
    ) -> InferenceExpType:
        """Inference method."""
        self.restore()
        prediction_data = data.copy()
        model = load_pkl(self.checkpoint_directory / "model.pkl")
        prediction_data[self.prediction_columns_name[0]] = model.predict(
            prediction_data, with_label=False
        )

        if save_df:
            prediction_data[self.columns_to_save(prediction_data)].to_csv(
                self.prediction_directory / (file_name + ".csv"), index=False
            )
        del model
        empty_cache()
        return prediction_data

    def eval_exp(self, comparison_score_metrics: Optional[pd.DataFrame] = None) -> EvalExpType:
        """Evaluate method."""
        self.predict()
        results = self._parse_metrics_to_data_frame(self.evaluator.get_evals())
        if isinstance(comparison_score_metrics, pd.DataFrame):
            self.plot_comparison_score_vs_predictions(
                comparison_score_metrics=comparison_score_metrics, predictions_metrics=results
            )
        (
            best_validation_scores,
            best_test_scores,
            best_prediction_name,
        ) = self.evaluator.get_experiment_best_scores(
            results=results,
            experiment_name=self.experiment_name,
            model_type=self.model_type,
            features_name=self.configuration["features"],
        )
        self._save_prediction_name_selector(best_prediction_name)
        return {"validation": best_validation_scores, "test": best_test_scores}

    def plot_comparison_score_vs_predictions(
        self, comparison_score_metrics: pd.DataFrame, predictions_metrics: pd.DataFrame
    ) -> None:
        """Process metrics data and plot scores of the comparison score and the predictions."""
        comparison_score = comparison_score_metrics[
            comparison_score_metrics.experiments == SINGLE_MODEL_NAME
        ]
        comparison_score_test = comparison_score_metrics[
            comparison_score_metrics.experiments == "test"
        ]

        comparison_score_test.iloc[
            :, comparison_score_test.columns.get_loc("experiments")
        ] = SINGLE_MODEL_NAME
        comparison_score = pd.concat([comparison_score, comparison_score_test])
        predictions_metrics["experiments"] = SINGLE_MODEL_NAME
        predictions_metrics["type"] = "IG_model"
        predictions_metrics = predictions_metrics[comparison_score.columns]
        scores = pd.concat([predictions_metrics, comparison_score])
        self.plotting_summary_scores(scores)

    def _parse_metrics_to_data_frame(
        self, eval_metrics: MetricsEvalType, file_name: str = "results"
    ) -> pd.DataFrame:
        """Convert eval metrics from dict format to pandas dataframe."""
        total_evals = []
        for data_name in eval_metrics:
            evals = eval_metrics[data_name][self.prediction_columns_name[0]]
            save_yml(evals, self.eval_directory / f"{data_name}_{file_name}metrics.yaml")
            global_evals = evals["global"]
            global_evals["prediction"] = self.prediction_columns_name[0]
            global_evals["split"] = data_name
            total_evals.append(global_evals)
        results = pd.DataFrame(total_evals).sort_values(["prediction", "split"])
        results.to_csv((self.eval_directory / f"{file_name}.csv"), index=False)
        return results
