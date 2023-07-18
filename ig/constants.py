"""Used to declare all the constant variable."""
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

ExpPredictTypeBase = Optional[Tuple[pd.DataFrame, pd.DataFrame]]
EvalExpTypeBase = Optional[Dict[str, pd.DataFrame]]
InferenceExpTypeBase = Optional[pd.DataFrame]
ExpPredictType = Tuple[pd.DataFrame, pd.DataFrame]
EvalExpType = Dict[str, pd.DataFrame]
InferenceExpType = pd.DataFrame

MetricsEvalType = Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
TuningOptParamsType = Tuple[Dict[str, Any], Dict[str, Any]]
TuneResults = Dict[str, Union[str, float]]
