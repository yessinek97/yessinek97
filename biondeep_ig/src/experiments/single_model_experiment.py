"""Classes to launch training using a single model."""
import pandas as pd

from biondeep_ig.src import Evals
from biondeep_ig.src import ID_NAME
from biondeep_ig.src.experiments.base import BaseExperiment
from biondeep_ig.src.utils import load_pkl
from biondeep_ig.src.utils import save_yml


class SingleModel(BaseExperiment):
    """Class to handle training a single specific model type."""

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
        """Initialize the single model experiment."""
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

        self.validation_column = kwargs["validation_column"]
        self.plot_shap_values = kwargs.get("plot_shap_values", False)
        self.validation_split_path = self.splits_path / (
            f'validation_split_{self.configuration["processing"]["seed"]}.csv'
        )
        self.load_data_set()
        if self.train_data_path:
            self.initialize_checkpoint_directory()
            if self.validation_split_path.exists():
                validation_split = pd.read_csv(self.validation_split_path)
                self.train_data = self.train_data.data.merge(
                    validation_split, on=[ID_NAME], how="left"
                )
            else:
                self.train_data.train_val_split()
                self.train_data = self.train_data.data
                self.train_data[[ID_NAME, self.validation_column]].to_csv(
                    self.validation_split_path, index=False
                )

            self.validation = self.train_data[self.train_data[self.validation_column] == 1]
            self.train_data = self.train_data[self.train_data[self.validation_column] == 0]

    @property
    def prediction_columns_name(self):
        """Return prediction_columns_name variable."""
        return ["prediction"]

    def train(self):
        """Training method."""
        model = self.single_fit(
            self.train_data,
            self.validation,
            self.checkpoint_directory,
            prediction_name=self.prediction_columns_name[0],
        )
        return model

    def predict(self, save_df=True):
        """Predict method."""
        test_data = self.inference(self.test_data, save_df=save_df, file_name="test")
        validation = self.inference(self.validation, save_df=save_df, file_name="validation")
        train_data = self.inference(self.train_data, save_df=save_df, file_name="train")
        return train_data, validation, test_data

    def inference(self, data, save_df=False, file_name=""):
        """Inference method."""
        self.restore()
        prediction_data = data.copy()
        model = load_pkl(self.checkpoint_directory / "model.pkl")
        prediction_data[self.prediction_columns_name[0]] = model.predict(
            prediction_data, with_label=False
        )

        if save_df:
            prediction_data.to_csv(self.prediction_directory / (file_name + ".csv"), index=False)
        return prediction_data

    def eval_exp(self):
        """Evaluate method."""
        self.predict()
        results = self._parse_metrics_to_data_frame(self.evaluator.get_evals())
        best_validation_scores, best_test_scores = self.evaluator.get_experiment_best_scores(
            results=results,
            experiment_name=self.experiment_name,
            model_type=self.model_type,
            features_name=self.configuration["features"],
        )
        return {"validation": best_validation_scores, "test": best_test_scores}

    def _parse_metrics_to_data_frame(
        self, eval_metrics: Evals, file_name: str = "results"
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
