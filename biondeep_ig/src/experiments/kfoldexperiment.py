"""Classes to launch training using Kfold strategy."""
import numpy as np
import pandas as pd

from biondeep_ig.src import Evals
from biondeep_ig.src import ID_NAME
from biondeep_ig.src.experiments.base import BaseExperiment
from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.utils import load_pkl
from biondeep_ig.src.utils import maybe_int
from biondeep_ig.src.utils import save_yml

log = get_logger("Kfold")


class KfoldExperiment(BaseExperiment):
    """Class to handle training with Kfold strategy."""

    def __init__(
        self,
        train_data_path,
        test_data_path,
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
            train_data_path=train_data_path,
            test_data_path=test_data_path,
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

        self.validation_split_path = self.splits_path / (
            (
                f'kfold_split_{self.configuration["processing"]["fold"]}_'
                f'{self.configuration["processing"]["seed"]}.csv'
            )
        )

        self.load_data_set()
        if self.train_data_path:
            self.initialize_checkpoint_directory()
            if self.validation_strategy:
                if self.validation_split_path.exists():
                    validation_split = pd.read_csv(self.validation_split_path)
                    self.train_data = self.train_data.data.merge(
                        validation_split, on=[ID_NAME], how="left"
                    )
                else:
                    self.train_data.kfold_split()
                    self.train_data = self.train_data.data
                    self.train_data[[ID_NAME, self.split_column]].to_csv(
                        self.validation_split_path, index=False
                    )
            else:
                self.train_data = self.train_data.data
            if self.split_column not in self.train_data.columns:
                raise KeyError(f"{self.split_column} column is missing")

    @property
    def prediction_columns_name(self):
        """Return prediction columns name variable."""
        return [f"prediction_{split}" for split in range(self.configuration["processing"]["fold"])]

    @property
    def kfold_prediction_name(self):
        """Return kfold prediction columns name."""
        return [f"prediction_{operation}" for operation in self.kfold_operations]

    def train(self):
        """Training method."""
        models = self.multiple_fit(train=self.train_data, split_column=self.split_column)
        return models

    def predict(self, save_df=True):
        """Predict method."""
        test_data = self.inference(self.test_data, save_df, file_name="test")

        train_data = []
        for model_path in self.checkpoint_directory.iterdir():
            split = maybe_int(model_path.name.replace("split_", ""))
            model = load_pkl(model_path / "model.pkl")

            data = self.train_data[self.train_data[self.split_column] == split].copy()
            data["prediction"] = model.predict(data, with_label=False)

            train_data.append(data)
        train_data = pd.concat(train_data)

        if save_df:
            train_data.to_csv(self.prediction_directory / "train.csv")
        return train_data, test_data

    def inference(self, data, save_df=False, file_name=""):
        """Inference method."""
        self.restore()
        prediction_data = data.copy()
        for model_path in self.checkpoint_directory.iterdir():
            split = model_path.name.replace("split_", "")
            model = load_pkl(model_path / "model.pkl")
            prediction_data[f"prediction_{split}"] = model.predict(data, with_label=False)

        for operation in self.kfold_operations:
            prediction_data[f"prediction_{operation}"] = getattr(np, operation)(
                prediction_data[self.prediction_columns_name], axis=1
            )

        if save_df:
            prediction_data.to_csv(self.prediction_directory / (file_name + ".csv"), index=False)
        return prediction_data

    def eval_exp(self):
        """Evaluate method."""
        _, test_data = self.predict()
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
