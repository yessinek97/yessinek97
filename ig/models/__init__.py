"""Import all the needed model classes from models package."""
from typing import Union

from ig.models.cat_boost_model import CatBoostModel  # noqa
from ig.models.label_propagation_model import LabelPropagationModel  # noqa
from ig.models.lgbm_model import LgbmModel  # noqa
from ig.models.llm_mixed_model import LLMMixedModel  # noqa
from ig.models.llm_model import LLMModel  # noqa
from ig.models.logistic_regression_model import LogisticRegressionModel  # noqa
from ig.models.random_forest_model import RandomForestModel  # noqa
from ig.models.svm_model import SupportVectorMachineModel  # noqa
from ig.models.xgboost_model import XgboostModel  # noqa

BaseModelType = Union[
    XgboostModel,
    LgbmModel,
    CatBoostModel,
    LabelPropagationModel,
    LogisticRegressionModel,
    RandomForestModel,
    SupportVectorMachineModel,
    LLMModel,
    LLMMixedModel,
]
