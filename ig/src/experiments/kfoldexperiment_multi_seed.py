"""Classes to launch training using Kfold strategy."""
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ig import DEFAULT_SEED, KFOLD_MODEL_NAME
from ig.constants import EvalExpType, ExpPredictType, InferenceExpType, MetricsEvalType
from ig.src.dataset import Dataset
from ig.src.experiments.base import BaseExperiment
from ig.src.logger import get_logger
from ig.src.models import BaseModel, TrainMultiSeedKfold
from ig.src.utils import load_pkl, maybe_int, plotting_kfold_shap_values, save_yml

log: Logger = get_logger("KfoldMultiSeed")


class KfoldMultiSeedExperiment(BaseExperiment):
    """Class to handle training with Multi seed Kfold strategy."""

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
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the Multi seed KFold experiment."""
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            unlabeled_path=unlabeled_path,
            configuration=configuration,
            experiment_name=experiment_name,
            folder_name=folder_name,
            sub_folder_name=sub_folder_name,
            experiment_directory=experiment_directory,
            features_configuration_path=features_configuration_path,
        )

        self.split_column = str(kwargs["split_column"])
        self.plot_shap_values = bool(kwargs.get("plot_shap_values", False))
        self.plot_kfold_shap_values = bool(kwargs.get("plot_kfold_shap_values", False))
        self.kfold_operations = list(kwargs.get("operation", ["mean"]))
        self.statistics_opts = list(kwargs.get("statistics", ["mean"]))
        self.nbr_seeds = int(str(kwargs.get("nbr_seeds", 2)))
        self.model_seed = bool(kwargs.get("model_seed", False))
        self.split_seed = bool(kwargs.get("split_seed", False))
        self.print_seeds_evals = bool(kwargs.get("print_seeds_evals", False))
        self.save_seed_predictions = bool(kwargs.get("save_seed_predictions", False))
        self.seed_plot_fig = bool(kwargs.get("seed_plot_fig", False))
        self.seeds = [int(seed) for seed in kwargs.get("seeds", [DEFAULT_SEED])]
        self.evaluator.set_is_plot_fig(self.seed_plot_fig)
        self.sub_model_directory_names: List[str] = []
        self.seeds_prediction_columns_name: List[str] = []
        self.train_predictions: Dict[str, pd.DataFrame] = {}
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

    @property
    def n_fold(self) -> int:
        """Return fold number for kfold split."""
        return int(self.configuration["processing"].get("fold", 5))

    def train(self) -> TrainMultiSeedKfold:
        """Training method."""
        models_seeds: TrainMultiSeedKfold = {}
        shap_values: List[pd.DataFrame] = []
        self.train_data.force_validation_strategy()
        self.evaluator.print_evals = self.print_seeds_evals
        for seed in self.seeds:
            sub_model_directory = f"{seed}"
            log.info(" seed %s", seed)
            if self.split_seed:
                self.train_data.kfold_split(self.split_column, seed)
                sub_model_directory += "_split"
            if self.model_seed:
                model_params = self.model_configuration["model_params"]
                model_params["seed"] = seed
                self.model_configuration["model_params"] = model_params
                sub_model_directory += "_model"
            self.sub_model_directory_names.append(sub_model_directory)
            models = self.multiple_fit(
                train=self.train_data(),
                split_column=self.split_column,
                sub_model_directory=sub_model_directory,
                multi_seed=True,
            )
            models_seeds[sub_model_directory] = models
            self.train_predictions[sub_model_directory] = self.seed_predict(
                sub_model_directory, save_df=self.save_seed_predictions
            )
            self.configuration["sub_model_directory_names"] = self.sub_model_directory_names
            save_yml(self.configuration, self.experiment_directory / "configuration.yml")
            if self.plot_shap_values:
                shap_values.append(
                    pd.read_csv(self.eval_directory / f"shap_values_{sub_model_directory}.csv")
                )

        if self.plot_shap_values and self.plot_kfold_shap_values:
            shap_values_df = pd.concat(shap_values)
            shap_values_df = (
                shap_values_df.groupby("features").scores.mean().rename("scores").reset_index()
            )
            plotting_kfold_shap_values(shap_values_df, self.eval_directory / "shap.png")

        self.evaluator.reset_print_evals()
        return models_seeds

    def predict(self, save_df: bool = True) -> ExpPredictType:
        """Predict method."""
        train_data = self.train_data().copy()
        for sub_model_directory_name in self.sub_model_directory_names:
            prediction = self.train_predictions[sub_model_directory_name]
            prediction.rename(
                columns={"prediction": f"prediction_{sub_model_directory_name}"}, inplace=True
            )
            train_data = train_data.merge(
                prediction[[self.train_data.id_column, f"prediction_{sub_model_directory_name}"]],
                on=self.train_data.id_column,
            )
        seed_prediction_columns_name = [
            f"prediction_{sub_model_directory_name}"
            for sub_model_directory_name in self.sub_model_directory_names
        ]
        train_data["prediction"] = train_data[seed_prediction_columns_name].mean(axis=1)
        train_data.drop([self.split_column], axis=1, inplace=True)
        if save_df:
            train_data[
                self.columns_to_save(train_data, is_train=True) + seed_prediction_columns_name
            ].to_csv(self.prediction_directory / "train.csv", index=False)

        test_data = self.inference(self.test_data(), True, "test")
        return train_data, test_data

    def inference(
        self, data: pd.DataFrame, save_df: bool = True, file_name: str = ""
    ) -> InferenceExpType:
        """Inference method."""
        test_data = data.copy()
        if not self.prediction_columns_name:
            self.prediction_columns_name = [f"prediction_{split}" for split in range(self.n_fold)]
        if not self.sub_model_directory_names:
            self.sub_model_directory_names = self.configuration["sub_model_directory_names"]

        for sub_model_directory_name in self.sub_model_directory_names:
            data, prediction_columns_name = self.inference_seed(
                test_data,
                sub_model_directory_name,
                save_df=self.save_seed_predictions,
                file_name=file_name,
            )
            test_data = test_data.merge(
                data[[self.id_column] + prediction_columns_name],
                how="left",
                on=self.id_column,
            )
        for operation in self.kfold_operations:
            test_data[f"prediction_{operation}"] = test_data[
                [col for col in self.seeds_prediction_columns_name if col.endswith(operation)]
            ].mean(axis=1)

        for split in range(self.n_fold):
            test_data[f"prediction_{split}"] = test_data[
                [
                    f"prediction_{sub_model_directory_name}_{split}"
                    for sub_model_directory_name in self.sub_model_directory_names
                ]
            ].mean(axis=1)
            self.seeds_prediction_columns_name.append(f"prediction_{split}")
        if save_df:
            test_data[self.columns_to_save(test_data, is_train=False)].to_csv(
                self.prediction_directory / f"{file_name}.csv", index=False
            )
        return test_data

    def seed_predict(self, sub_model_directory_name: str, save_df: bool) -> pd.DataFrame:
        """Return the predictions for each seed."""
        train_data: List[pd.DataFrame] = []
        checkpoint_directory = self.checkpoint_directory / sub_model_directory_name

        for model_path in checkpoint_directory.iterdir():
            split = maybe_int(model_path.name.replace("split_", ""))
            model: BaseModel = load_pkl(model_path / "model.pkl")
            data = self.train_data().copy()
            data = data[data[self.split_column] == split].copy()
            data["prediction"] = model.predict(data, with_label=False)

            train_data.append(data)
        train_data_df = pd.concat(train_data)

        if save_df:
            train_data_df[
                self.columns_to_save(train_data_df, is_train=True) + [self.split_column]
            ].to_csv(
                self.prediction_directory / f"train_{sub_model_directory_name}.csv", index=False
            )

        return train_data_df

    def inference_seed(
        self,
        data: pd.DataFrame,
        sub_model_directory_name: str,
        save_df: bool = False,
        file_name: str = "",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Inference method."""
        self.restore()
        prediction_data = data.copy()
        prediction_columns_name: List[str] = []
        checkpoint_directory = self.checkpoint_directory / sub_model_directory_name
        for model_path in checkpoint_directory.iterdir():
            split = model_path.name.replace("split_", "")
            model = load_pkl(model_path / "model.pkl")
            prediction_data[f"prediction_{sub_model_directory_name}_{split}"] = model.predict(
                data, with_label=False
            )
            prediction_columns_name.append(f"prediction_{sub_model_directory_name}_{split}")
        for operation in self.kfold_operations:
            prediction_data[f"prediction_{sub_model_directory_name}_{operation}"] = getattr(
                np, operation
            )(prediction_data[prediction_columns_name], axis=1)
            prediction_columns_name.append(f"prediction_{sub_model_directory_name}_{operation}")
        if save_df:
            prediction_data[
                self.columns_to_save(prediction_data, multi_seed=True, is_train=False)
                + prediction_columns_name
            ].to_csv(
                self.prediction_directory / f"{file_name}_{sub_model_directory_name}.csv",
                index=False,
            )
        self.seeds_prediction_columns_name.extend(prediction_columns_name)
        return prediction_data, prediction_columns_name

    def eval_exp(self, comparison_score_metrics: Optional[pd.DataFrame]) -> EvalExpType:
        """Evaluate method."""
        validation_data, test_data = self.predict()
        log.info("-Kfold Multi seed predictions")
        log.info(" - Average fold: ")
        for split in range(self.n_fold):

            log.info("   - fold %s :", split)
            self.evaluator.compute_metrics(
                data=test_data,
                prediction_name=f"prediction_{split}",
                data_name="test",
            )

        self.evaluator.print_evals = self.print_seeds_evals
        for sub_model_directory_name in self.sub_model_directory_names:
            if self.print_seeds_evals:
                log.info("    - Seed: %s ", sub_model_directory_name)
            for operation in self.kfold_operations:
                if self.print_seeds_evals:
                    log.info("     - %s :", operation)
                column_name = f"prediction_{sub_model_directory_name}"
                new_column_name = f"prediction_{sub_model_directory_name}_{operation}"
                self.evaluator.compute_metrics(
                    data=validation_data.rename(columns={column_name: new_column_name}),
                    prediction_name=f"prediction_{sub_model_directory_name}_{operation}",
                    data_name="validation",
                )
                self.evaluator.compute_metrics(
                    data=test_data,
                    prediction_name=f"prediction_{sub_model_directory_name}_{operation}",
                    data_name="test",
                )
        self.evaluator.reset_print_evals()

        log.info(" - Ensembling: ")
        self.evaluator.set_is_plot_fig(True)
        for operation in self.kfold_operations:
            if self.evaluator.print_evals:
                log.info("     - %s :", operation)

            self.evaluator.compute_metrics(
                data=validation_data.rename(columns={"prediction": f"prediction_{operation}"}),
                prediction_name=f"prediction_{operation}",
                data_name="validation",
            )
            self.evaluator.compute_metrics(
                data=test_data, prediction_name=f"prediction_{operation}", data_name="test"
            )
        self.evaluator.set_is_plot_fig(self.seed_plot_fig)
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
            results=results[
                results.prediction.isin(self.prediction_columns_name + self.kfold_prediction_name)
            ],
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
        self.get_statistic_seeds_scores()
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
        predictions_metrics.to_csv(self.eval_directory / "predictions_metrics.csv", index=False)
        predictions_metrics = predictions_metrics[
            np.logical_not(
                (predictions_metrics.prediction == self.evaluator.prediction_name_selector)
                & (predictions_metrics.split == "validation")
            )
        ]
        scores = pd.concat([predictions_metrics, comparison_score])
        self.plotting_summary_scores(scores)

    def _parse_metrics_to_data_frame(
        self, eval_metrics: MetricsEvalType, file_name: str = "results"
    ) -> pd.DataFrame:
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

    def get_statistic_seeds_scores(self) -> None:
        """Compute and display statistic for the metrics selector score cross different seed."""
        if "mean" in self.kfold_operations:
            test_scores = self.evaluator.get_evals()["test"]
            seeds_metrics = [
                test_scores[f"prediction_{x}_mean"]["global"][self.evaluator.metric_selector]
                for x in self.sub_model_directory_names
            ]
            metrics = {
                "mean": float(round(np.mean(seeds_metrics), 4)),
                "min": float(round(np.min(seeds_metrics), 4)),
                "max": float(round(np.max(seeds_metrics), 4)),
                "std": float(round(np.std(seeds_metrics), 4)),
            }
            log.info("#############")
            log.info("Statistic report across seeds: ")
            log.info(" -mean: %s", metrics["mean"])
            log.info(" -min:  %s", metrics["min"])
            log.info(" -max : %s", metrics["max"])
            log.info(" -std : %s", metrics["std"])
            log.info("#############")
            save_yml(metrics, self.experiment_directory / "seeds_report.yml")
