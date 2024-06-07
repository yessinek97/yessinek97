"""Classes to launch training using double Kfold strategy."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ig.constants import EvalExpType, ExpPredictType, InferenceExpType, MetricsEvalType
from ig.cross_validation.base import BaseExperiment
from ig.dataset.dataset import Dataset
from ig.src.logger import get_logger
from ig.src.utils import load_pkl, maybe_int, save_yml
from ig.utils.torch_helper import empty_cache

log = get_logger("DoubleKfold")


class DoubleKfold(BaseExperiment):
    """Class to handle training with double Kfold strategy."""

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
        """Initialize the single double KFold experiment."""
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
        self.split_column = str(kwargs["split_column"])
        self.plot_shap_values = bool(kwargs.get("plot_shap_values", False))
        self.plot_kfold_shap_values = bool(kwargs.get("plot_kfold_shap_values", False))
        self.kfold_operations = list(kwargs.get("operation", ["mean"]))
        self.statistics_opts = list(kwargs.get("statistics", ["mean"]))
        if isinstance(self.train_data, Dataset):
            self.initialize_checkpoint_directory()
            if self.split_column not in self.train_data().columns:
                raise KeyError(f"{self.split_column} column is missing")
            self.prediction_columns_name = [
                f"prediction_{split}" for split in self.train_data()[self.split_column].unique()
            ]
        self.kfold_prediction_name = [
            f"prediction_{operation}" for operation in self.kfold_operations
        ]

    def train(self) -> None:
        """Training method."""
        for split in np.sort(self.train_data()[self.split_column].unique()):
            log.info("          ----------------------")
            log.info("          Begin Split %s :", split)
            train = self.train_data().copy()
            train = train[train[self.split_column] != split]
            self.multiple_fit(
                train=train,
                split_column=self.split_column,
                sub_model_directory=f"split_{split}",
            )
            log.info("          End Split %s :", split)
            log.info("          ----------------------")

    def predict(self, save_df: bool = True) -> ExpPredictType:
        """Predict method."""
        test_data = self.inference(self.test_data(), save_df, file_name="test")
        train_data: List[str] = []
        for sub_model_paths in self.checkpoint_directory.iterdir():
            split = maybe_int(sub_model_paths.name.replace("split_", ""))
            data = self.train_data().copy()
            data = data[data[self.split_column] == split].copy()
            data["prediction"] = 0
            nfold = 0
            for model_path in sub_model_paths.iterdir():
                nfold += 1
                model = load_pkl(model_path / "model.pkl")
                data["prediction"] += model.predict(data, with_label=False)
                del model
                empty_cache()
            data["prediction"] /= nfold
            train_data.append(data)
        train_data_df = pd.concat(train_data)

        if save_df:
            train_data_df[self.columns_to_save(train_data_df, is_train=True)].to_csv(
                self.prediction_directory / "train.csv"
            )
        return train_data_df, test_data

    def inference(
        self, data: pd.DataFrame, save_df: bool = False, file_name: str = ""
    ) -> InferenceExpType:
        """Inference method."""
        self.restore()
        prediction_data = data.copy()
        prediction_columns_name = []
        for sub_model_paths in self.checkpoint_directory.iterdir():
            sub_model_split = sub_model_paths.name.replace("split_", "")
            prediction_data[f"prediction_{sub_model_split}"] = 0
            nfold = 0
            for model_path in sub_model_paths.iterdir():
                nfold += 1
                model = load_pkl(model_path / "model.pkl")

                prediction_data[f"prediction_{sub_model_split}"] += model.predict(
                    data, with_label=False
                )

                del model
                empty_cache()
            prediction_data[f"prediction_{sub_model_split}"] /= nfold
            prediction_columns_name.append(f"prediction_{sub_model_split}")

        self.prediction_columns_name = prediction_columns_name
        for operation in self.kfold_operations:
            prediction_data[f"prediction_{operation}"] = getattr(np, operation)(
                prediction_data[self.prediction_columns_name], axis=1
            )

        if save_df:
            prediction_data[self.columns_to_save(prediction_data, is_train=False)].to_csv(
                self.prediction_directory / (file_name + ".csv"), index=False
            )
        return prediction_data

    def eval_exp(self, comparison_score_metrics: Optional[pd.DataFrame] = None) -> EvalExpType:
        """Evaluate method."""
        validation_data, test_data = self.predict()
        if self.evaluator.print_evals:
            for pred_name in self.prediction_columns_name:
                log.info("           -%s :", pred_name)
                self.evaluator.compute_metrics(
                    data=test_data,
                    prediction_name=pred_name,
                    split_name="test",
                    dataset_name=self.dataset_name,
                )
            log.info("            -Double Kfold predictions columns")

        for operation in self.kfold_operations:
            if self.evaluator.print_evals:
                log.info("             %s :", operation)
            self.evaluator.compute_metrics(
                data=test_data,
                prediction_name=f"prediction_{operation}",
                split_name="test",
                dataset_name=self.dataset_name,
            )
            self.evaluator.compute_metrics(
                data=validation_data.rename(columns={"prediction": f"prediction_{operation}"}),
                prediction_name=f"prediction_{operation}",
                split_name="validation",
                dataset_name=self.dataset_name,
            )
        results = self._parse_metrics_to_data_frame(eval_metrics=self.evaluator.get_evals())
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

        statistic_scores = self.evaluator.get_statistic_kfold_scores(
            statistic_opt=self.statistics_opts,
            prediction_columns_name=self.prediction_columns_name,
            experiment_name=self.experiment_name,
            model_type=self.model_type,
            features_name=self.configuration["features"],
        )
        self._save_prediction_name_selector(best_prediction_name)
        return {
            "validation": best_validation_scores,
            "test": best_test_scores,
            "statistic": statistic_scores,
        }

    def plot_comparison_score_vs_predictions(
        self, comparison_score_metrics: pd.DataFrame, predictions_metrics: pd.DataFrame
    ) -> None:
        """Process metrics data and plot scores of the comparison score and the predictions."""

    def _parse_metrics_to_data_frame(
        self, eval_metrics: MetricsEvalType, file_name: str = "results"
    ) -> pd.DataFrame:
        """Convert eval metrics from dict format to pandas dataframe."""
        total_evals = []
        for split_name in eval_metrics:
            evals_per_split_name = eval_metrics[split_name]
            save_yml(
                evals_per_split_name,
                self.eval_directory / f"{split_name}_{file_name}_metrics.yaml",
            )
            for prediction_name in evals_per_split_name:
                evals = evals_per_split_name[prediction_name]["global"]
                evals["prediction"] = prediction_name
                evals["split"] = split_name
                total_evals.append(evals)
        results = pd.DataFrame(total_evals).sort_values(["prediction", "split"])
        results.to_csv((self.eval_directory / f"{file_name}.csv"), index=False)
        return results
