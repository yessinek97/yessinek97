"""Classes to launch training using Kfold strategy."""
import numpy as np
import pandas as pd

from biondeep_ig import Evals
from biondeep_ig import KFOLD_MODEL_NAME
from biondeep_ig.src.experiments.base import BaseExperiment
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.processing_v1 import Dataset
from biondeep_ig.src.utils import load_pkl
from biondeep_ig.src.utils import maybe_int
from biondeep_ig.src.utils import save_yml

log = get_logger("Kfold")


class KfoldExperiment(BaseExperiment):
    """Class to handle training with Kfold strategy."""

    def __init__(
        self,
        train_data,
        test_data,
        configuration,
        folder_name,
        experiment_name=None,
        sub_folder_name=None,
        unlabeled_path=None,
        experiment_directory=None,
        **kwargs,
    ):
        """Initialize the KFold experiment."""
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            unlabeled_path=unlabeled_path,
            configuration=configuration,
            experiment_name=experiment_name,
            folder_name=folder_name,
            sub_folder_name=sub_folder_name,
            experiment_directory=experiment_directory,
        )

        self.split_column = kwargs["split_column"]
        self.plot_shap_values = kwargs.get("plot_shap_values", False)
        self.plot_kfold_shap_values = kwargs.get("plot_kfold_shap_values", False)
        self.kfold_operations = kwargs.get("operation", ["mean"])
        self.statistics_opts = kwargs.get("statistics", ["mean"])
        if isinstance(self.train_data, Dataset):
            self.initialize_checkpoint_directory()
            if self.split_column not in self.train_data().columns:
                raise KeyError(f"{self.split_column} column is missing")
            self.prediction_columns_name = [
                f"prediction_{split}" for split in self.train_data()[self.split_column].unique()
            ]

    @property
    def kfold_prediction_name(self):
        """Return kfold prediction columns name."""
        return [f"prediction_{operation}" for operation in self.kfold_operations]

    def train(self):
        """Training method."""
        models = self.multiple_fit(train=self.train_data(), split_column=self.split_column)
        return models

    def predict(self, save_df=True):
        """Predict method."""
        test_data = self.inference(self.test_data(), save_df, file_name="test")

        train_data = []
        for model_path in self.checkpoint_directory.iterdir():
            split = maybe_int(model_path.name.replace("split_", ""))
            model = load_pkl(model_path / "model.pkl")
            data = self.train_data().copy()
            data = data[data[self.split_column] == split].copy()
            data["prediction"] = model.predict(data, with_label=False)

            train_data.append(data)
        train_data = pd.concat(train_data)

        if save_df:
            train_data[self.columns_to_save(is_train=True)].to_csv(
                self.prediction_directory / "train.csv", index=False
            )
        return train_data, test_data

    def inference(self, data, save_df=False, file_name=""):
        """Inference method."""
        self.restore()
        prediction_data = data.copy()
        prediction_columns_name = []
        for model_path in self.checkpoint_directory.iterdir():
            split = model_path.name.replace("split_", "")
            model = load_pkl(model_path / "model.pkl")
            prediction_data[f"prediction_{split}"] = model.predict(data, with_label=False)
            prediction_columns_name.append(f"prediction_{split}")

        self.prediction_columns_name = prediction_columns_name

        for operation in self.kfold_operations:
            prediction_data[f"prediction_{operation}"] = getattr(np, operation)(
                prediction_data[self.prediction_columns_name], axis=1
            )

        if save_df:
            prediction_data[self.columns_to_save(is_train=False)].to_csv(
                self.prediction_directory / (file_name + ".csv"), index=False
            )
        return prediction_data

    def eval_exp(self, comparison_score_metrics):
        """Evaluate method."""
        validation_data, test_data = self.predict()
        if self.evaluator.print_evals:
            log.info("           -Kfold predictions")
        for operation in self.kfold_operations:
            if self.evaluator.print_evals:
                log.info(f"             {operation} :")
            self.evaluator.compute_metrics(
                data=validation_data.rename(columns={"prediction": f"prediction_{operation}"}),
                prediction_name=f"prediction_{operation}",
                data_name="validation",
            )
            self.evaluator.compute_metrics(
                data=test_data, prediction_name=f"prediction_{operation}", data_name="test"
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
        self, comparison_score_metrics, predictions_metrics: pd.DataFrame
    ):
        """Process metrics data and plot scores of the comparison score and the predications."""
        kfold_columns = self.prediction_columns_name + [self.evaluator.prediction_name_selector]
        comparison_score = comparison_score_metrics[
            comparison_score_metrics.experiments == KFOLD_MODEL_NAME
        ]
        comparison_score_test = comparison_score_metrics[
            comparison_score_metrics.experiments == "test"
        ]
        comparison_score = comparison_score[
            comparison_score.prediction.isin(self.prediction_columns_name)
        ]
        comparison_score_test.iloc[
            :, comparison_score_test.columns.get_loc("experiments")
        ] = KFOLD_MODEL_NAME
        comparison_score_test = pd.concat(
            [comparison_score_test for i in range(len(self.prediction_columns_name) + 1)]
        )
        comparison_score_test["prediction"] = kfold_columns
        comparison_score = pd.concat([comparison_score, comparison_score_test])
        predictions_metrics["experiments"] = KFOLD_MODEL_NAME
        predictions_metrics["type"] = "IG_model"
        predictions_metrics = predictions_metrics[comparison_score.columns]

        predictions_metrics = predictions_metrics[
            predictions_metrics.prediction.isin(kfold_columns)
        ]
        predictions_metrics.to_csv("./predictions_metrics.csv", index=False)
        predictions_metrics = predictions_metrics[
            np.logical_not(
                (predictions_metrics.prediction == self.evaluator.prediction_name_selector)
                & (predictions_metrics.split == "validation")
            )
        ]
        scores = pd.concat([predictions_metrics, comparison_score])
        self.plotting_summary_scores(scores)

    def _parse_metrics_to_data_frame(self, eval_metrics: Evals, file_name: str = "results"):
        """Convert eval metrics from dict format to pandas dataframe."""
        total_evals = []
        for data_name in eval_metrics:
            evals_per_data_name = eval_metrics[data_name]
            save_yml(
                evals_per_data_name, self.eval_directory / f"{data_name}_{file_name}_metrics.yaml"
            )
            for prediction_name in evals_per_data_name:
                evals = evals_per_data_name[prediction_name]["global"]
                evals["prediction"] = prediction_name
                evals["split"] = data_name
                total_evals.append(evals)
        results = pd.DataFrame(total_evals).sort_values(["prediction", "split"])
        results.to_csv((self.eval_directory / f"{file_name}.csv"), index=False)
        return results
