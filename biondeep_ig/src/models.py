"""File that contains all the model that will be used in the experiment."""
import sys
import warnings
from abc import ABC
from abc import abstractmethod

import lightgbm as lgb
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation

from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import ModelLogWriter
from biondeep_ig.src.utils import save_as_pkl

log = get_logger("Train/model")
original_stdout = sys.__stdout__


class BaseModel(ABC):
    """Basic class Method."""

    def __init__(
        self,
        features,
        parameters,
        label_name,
        prediction_name,
        checkpoints,
        other_params,
        folder_name,
        experiment_name,
        model_type,
        save_model,
    ):
        """Init method."""
        self.features = features
        self.parameters = parameters
        self.label_name = label_name
        self.prediction_name = prediction_name
        self.checkpoints = checkpoints
        self.other_params = other_params
        self.folder_name = folder_name
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.model = None
        self.shap_values = None
        self.model_logger = None
        self.save_model = save_model
        if self.save_model:
            self.checkpoints.parent.mkdir(parents=True, exist_ok=True)
            self.model_logger = ModelLogWriter(self.checkpoints.parent / "model.log")

    @property
    def model_meta_data(self):
        """Model meta data."""
        return {
            "model": self.model,
            "model_params": self.parameters,
            "features": self.features,
            "checkpoints": self.checkpoints,
        }

    @abstractmethod
    def predict(self):
        """Prediction method."""
        pass

    @abstractmethod
    def fit(self):
        """Fit method."""
        pass

    def eval_model(self, data, data_name, evaluator):
        """Eval method."""
        data = data.copy()
        data[self.prediction_name] = self.predict(data)
        return evaluator.compute_metrics(
            data=data, prediction_name=self.prediction_name, data_name=data_name
        )

    def generate_shap_values(self, dataframe):
        """Generate shap values."""
        return shap.TreeExplainer(self.model).shap_values(dataframe)

    @abstractmethod
    def _create_matrix(self):
        """Return the correct data structure. Object that is required by the model."""
        pass


warnings.filterwarnings("ignore")


class XgboostModel(BaseModel):
    """Xgboost model class."""

    def __init__(
        self,
        features,
        parameters,
        label_name,
        prediction_name,
        checkpoints,
        other_params,
        folder_name,
        experiment_name,
        model_type,
        save_model,
    ):
        """Init."""
        super().__init__(
            features,
            parameters,
            label_name,
            prediction_name,
            checkpoints,
            other_params,
            folder_name,
            experiment_name,
            model_type,
            save_model,
        )

    def fit(self, train_data, val_data):
        """Fitting model."""
        dtrain = self._create_matrix(train_data)
        dval = self._create_matrix(val_data)

        self.model = xgb.train(
            params=self.parameters,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dval, "valid")],
            num_boost_round=self.other_params.get("num_boost_round", 20000),
            verbose_eval=self.other_params.get("verbose_eval", True),
            early_stopping_rounds=self.other_params.get("early_stopping_rounds", 20),
        )

        self.shap_values = self.generate_shap_values(train_data[self.features])

        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data, with_label=True):
        """Prediction method."""
        dmatrix = self._create_matrix(data, with_label)
        best_iteration = self.model.best_iteration
        return self.model.predict(dmatrix, ntree_limit=best_iteration)

    def _create_matrix(self, data, with_label=True):
        """Data model creation."""
        label = data[self.label_name] if with_label else None
        return xgb.DMatrix(data=data[self.features], label=label, feature_names=self.features)


class LgbmModel(BaseModel):
    """This is an implementation of lgbm model.Based on the Native Microsoft Implementation."""

    def __init__(
        self,
        features,
        parameters,
        label_name,
        prediction_name,
        checkpoints,
        other_params,
        folder_name,
        experiment_name,
        model_type,
        save_model,
    ):
        """Init."""
        super().__init__(
            features,
            parameters,
            label_name,
            prediction_name,
            checkpoints,
            other_params,
            folder_name,
            experiment_name,
            model_type,
            save_model,
        )

    def generate_shap_values(self, dataframe):
        """Generate shap values."""
        return shap.TreeExplainer(self.model).shap_values(dataframe)[1]

    def fit(self, train_data, val_data):
        """Fitting model."""
        # Load data
        true_data = train_data
        train_data = self._create_matrix(train_data)
        val_data = self._create_matrix(val_data, train_data, train=False)

        # train model
        self.model = lgb.train(
            self.parameters,
            train_data,
            num_boost_round=self.other_params.get("num_boost_round"),
            valid_sets=[val_data],
            verbose_eval=self.other_params["verbose_eval"],
            early_stopping_rounds=self.other_params.get("early_stopping_rounds"),
        )
        self.shap_values = self.generate_shap_values(true_data[self.features])
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data, with_label=True):
        """Prediction method."""
        return self.model.predict(data[self.features], num_iteration=self.model.best_iteration)

    def _create_matrix(self, data, x_data=None, train=True, with_label=True):
        """Data model creation."""
        label = data[self.label_name] if with_label else None
        if train:
            return lgb.Dataset(data[self.features], label=label, feature_name=self.features)

        else:
            return lgb.Dataset(data[self.features], reference=x_data)


class CatBoostModel(BaseModel):
    """This is an implementation of catboost model.Based on the Native Implementation."""

    def __init__(
        self,
        features,
        parameters,
        label_name,
        prediction_name,
        checkpoints,
        other_params,
        folder_name,
        experiment_name,
        model_type,
        save_model,
    ):
        """Init."""
        super().__init__(
            features,
            parameters,
            label_name,
            prediction_name,
            checkpoints,
            other_params,
            folder_name,
            experiment_name,
            model_type,
            save_model,
        )

    def fit(self, train, validation, with_label=True):
        """Fitting model."""
        # create model
        model = CatBoostClassifier(**self.parameters)
        # train model
        self.model = model.fit(
            X=train[self.features].values,
            y=train[self.label_name].values,
            eval_set=(validation[self.features].values, validation[self.label_name].values),
            log_cout=sys.stdout,
            log_cerr=sys.stderr,
        )
        self.shap_values = self.generate_shap_values(train[self.features])
        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data, with_label=True):
        """Prediction method."""
        return self.model.predict_proba(data[self.features].values)[:, 1]

    def _create_matrix(self, data, with_label=True):
        """Data model creation."""
        pass


class LabelPropagationModel(BaseModel):
    """Label Propagation model class."""

    def __init__(
        self,
        features,
        parameters,
        label_name,
        prediction_name,
        checkpoints,
        other_params,
        folder_name,
        experiment_name,
        model_type,
    ):
        """Init."""
        super().__init__(
            features,
            parameters,
            label_name,
            prediction_name,
            checkpoints,
            other_params,
            folder_name,
            experiment_name,
            model_type,
        )

    def fit(self, train_data, val_data):
        """Fitting model."""
        dtrain = self._create_matrix(train_data)
        labels = self._create_matrix(train_data, with_label=False)
        self.model = LabelPropagation(**self.parameters)
        self.model = self.model.fit(dtrain, labels)
        self.shap_values = None
        log.info(" Shap values for this model type not supported")

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data, with_label=True):
        """Prediction method."""
        dmatrix = self._create_matrix(data)
        return self.model.predict(dmatrix)

    def _create_matrix(self, data, with_label=True):
        """Data model creation."""
        if with_label:
            return data[self.features].to_numpy()
        else:
            return data[self.label_name].to_numpy()


class LogisticRegressionModel(BaseModel):
    """This is an implementation of LogisticRegression model."""

    """Based on the Native Implementation."""

    def __init__(
        self,
        features,
        parameters,
        label_name,
        prediction_name,
        checkpoints,
        other_params,
        folder_name,
        experiment_name,
        model_type,
        save_model,
    ):
        """Init."""
        super().__init__(
            features,
            parameters,
            label_name,
            prediction_name,
            checkpoints,
            other_params,
            folder_name,
            experiment_name,
            model_type,
            save_model,
        )

    def fit(self, train, validation, with_label=True):
        """Fitting model."""
        self.model = LogisticRegression().fit(train[self.features], train[self.label_name])

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data, with_label=True):
        """Prediction method."""
        return self.model.predict_proba(data[self.features].values)[:, 1]

    def _create_matrix(self, data, with_label=True):
        """Data model creation."""
        pass
