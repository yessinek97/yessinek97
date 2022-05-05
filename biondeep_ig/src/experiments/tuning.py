"""Class used to fine tuning the model parameters."""
import logging

import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt.fmin import fmin

from biondeep_ig import Evals
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig.src.experiments.base import BaseExperiment
from biondeep_ig.src.metrics import topk_global
from biondeep_ig.src.processing_v1 import Dataset
from biondeep_ig.src.utils import convert_int_params
from biondeep_ig.src.utils import save_as_pkl
from biondeep_ig.src.utils import save_yml

logger = logging.getLogger("Tuning")


class Tuning(BaseExperiment):
    """Tune model hyperparams Class."""

    def __init__(
        self,
        train_data,
        test_data,
        configuration,
        folder_name,
        experiment_param,
        is_compute_metrics,
        experiment_name=None,
        sub_folder_name=None,
        unlabeled_path=None,
        **kwargs,
    ):
        """Initialize the Tuning experiment."""
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            unlabeled_path=unlabeled_path,
            configuration=configuration,
            experiment_name=experiment_name,
            folder_name=folder_name,
            sub_folder_name=sub_folder_name,
            is_compute_metrics=is_compute_metrics,
        )
        self.experiment_directory = MODELS_DIRECTORY / folder_name / self.model_type
        self.plot_shap_values = kwargs.get("plot_shap_values", False)
        self.plot_kfold_shap_values = kwargs.get("plot_kfold_shap_values", False)
        self.initialize_checkpoint_directory(tuning_option=True)
        save_yml(self.configuration, self.experiment_directory / "configuration.yml")
        if experiment_name == "SingleModel":
            self._initialize_single_fit(experiment_param=experiment_param)
        if experiment_name == "KfoldExperiment":
            self._initialize_multi_fit(experiment_param=experiment_param)
        self.int_columns = []

        self.model_configuration = self.convert_model_params_to_hp(self.model_configuration)
        self.save_model = False
        self.evaluator.curve_plot_directory = None

    @property
    def nbr_trials(self):
        """Return nbr of trials attribute."""
        return self.configuration["tuning"]["nbr_trials"]

    @property
    def maximize(self):
        """Return maximize attribute."""
        return self.configuration["tuning"]["maximize"]

    def optimize_parms(self, train_func, model_param_spaces):
        """Optimize the model hyperparams."""
        trials = Trials()
        fmin(
            fn=train_func,
            space=model_param_spaces,
            algo=tpe.suggest,
            max_evals=self.nbr_trials,
            trials=trials,
            verbose=1,
        )
        best_index = np.argmin(trials.losses())
        results = trials.trials[best_index]["result"]
        return results, trials.trials

    def tune_single_model(self, model_configuration_spaces):
        """Optimize hyperparams for single model experiment."""

        def single_fit(model_configuration):
            """Optimize hyperparams for single model experiment."""
            model_configuration["model_params"] = convert_int_params(
                self.int_columns, model_configuration["model_params"]
            )
            self.configuration["model"] = model_configuration
            model = self.single_fit(
                train=self.train_split,
                validation=self.validation_split,
                checkpoints=None,
                model_name=None,
            )
            pred = model.predict(self.test_data(), with_label=False)
            score = topk_global(self.test_data("label"), pred)[0]
            if self.maximize:
                score *= -1
            return {"loss": score, "status": STATUS_OK, "params": model_configuration}

        return self.optimize_parms(single_fit, model_configuration_spaces)

    def tune_kfold_model(self, model_configuration_spaces):
        """Optimize hyperparams for Kfold experiment."""

        def multiple_fit(model_configuration):
            """Optimize hyperparams for Kfold experiment."""
            model_configuration["model_params"] = convert_int_params(
                self.int_columns, model_configuration["model_params"]
            )
            self.configuration["model"] = model_configuration
            models = self.multiple_fit(
                train=self.train_data(),
                split_column=self.split_column,
                sub_model_directory=None,
            )
            preds = []
            for _, i_model in models.items():
                preds.append(i_model.predict(self.test_data(), with_label=False))

            preds = np.mean(preds, axis=0)
            # TODO change topk_global to configurbale param
            score = topk_global(self.test_data("label"), preds)[0]
            if self.maximize:
                score *= -1
            return {"loss": score, "status": STATUS_OK, "params": model_configuration}

        return self.optimize_parms(multiple_fit, model_configuration_spaces)

    def convert_model_params_to_hp(self, model_param):
        """Convert model params to hp object."""
        model_configuration_dynamic = {}
        for key, value in zip(
            model_param["model_params"].keys(), model_param["model_params"].values()
        ):
            if isinstance(value, dict):
                model_configuration_dynamic[key] = hp.quniform(
                    key, value["low"], value["high"], value["q"]
                )
                if value["type"] == "int":
                    self.int_columns.append(key)
            else:
                model_configuration_dynamic[key] = value
        model_param["model_params"] = model_configuration_dynamic
        return model_param

    def train(self, features_list_path):  # pylint: disable=W0221
        """Train method."""
        if self.experiment_name == "SingleModel":
            results, all_trials = self.tune_single_model(
                model_configuration_spaces=self.model_configuration
            )

        elif self.experiment_name == "KfoldExperiment":
            results, all_trials = self.tune_kfold_model(
                model_configuration_spaces=self.model_configuration
            )
        else:
            raise NotImplementedError(
                (
                    "{experiment_name} is not defined; "
                    "choose from: [SingleModel, KfoldExperiment,"
                    "SingKfoldModel]"
                )
            )

        path = self.experiment_directory / self.sub_folder_name
        path.mkdir(exist_ok=True, parents=True)
        save_yml(results, path / f"{self.experiment_name}_{self.model_type}_best_model_params.yml")
        save_as_pkl(all_trials, path / f"{self.experiment_name}_{self.model_type}_trials.pkl")
        score = -1 * results["loss"] if self.maximize else results["loss"]
        return {
            "model": self.model_type,
            "experiment": self.experiment_name,
            "features": features_list_path,
            "score": score,
        }

    def eval_exp(self, comparison_score_metrics=None):
        """Eval method."""
        return

    def inference(self, data, save_df=False, file_name=""):
        """Inference method."""
        return

    def predict(self, save_df):
        """Predict method."""
        return

    def plot_comparison_score_vs_predictions(
        self, comparison_score_metrics, predictions_metrics: pd.DataFrame
    ):
        """Process metrics data and plot scores of the comparison score and the predications."""
        return

    def _initialize_single_fit(self, experiment_param):
        """Initialize data for single model."""
        self.validation_column = experiment_param["validation_column"]
        if isinstance(self.train_data, Dataset):

            if self.validation_column not in self.train_data().columns:
                raise KeyError(f"{self.validation_column} column is missing")
            self.validation_split = self.train_data()[
                self.train_data()[self.validation_column] == 1
            ]
            self.train_split = self.train_data()[self.train_data()[self.validation_column] == 0]

    def _initialize_multi_fit(self, experiment_param):
        """Initialize data for Kfold model."""
        self.split_column = experiment_param["split_column"]
        if isinstance(self.train_data, Dataset):
            if self.split_column not in self.train_data().columns:
                raise KeyError(f"{self.split_column} column is missing")

    def _parse_metrics_to_data_frame(self, eval_metrics: Evals, file_name: str = "results"):
        """Parse Metrics results from dictionary to dataframe object."""
