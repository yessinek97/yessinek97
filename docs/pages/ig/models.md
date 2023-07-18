## Configuration

All model related parameters can be changed in model configuration files. Defaults for each
available model are located at `configuration/model_config/<model>.yml`. The model configuration
file should be indicated under `models` in the
[global configuration file](./config.md#Global-configuration-file) to run the respective model per
experiment.

Multiple models can be defined in the
[global configuration file](./config.md#Global-configuration-file).

## Trainable models

### `XgboostModel`

Model based on the [XGBoost](https://xgboost.readthedocs.io/en/stable/) implementation of parallel
decision tree boosting.

### `LgbmModel`

Model based on the [LightGBM](https://lightgbm.readthedocs.io/en/latest/) implementation of decision
tree.

### `CatBoostModel`

Model based on the [CatBoost](https://catboost.ai/) implementation of gradient boosting on decision
trees

### `LogisticRegressionModel`

Model based on the
[Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
implementation of the logistic regression.

### `LabelPropagationModel`

Model based on the
[Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html)
implementation of the semi-supervised method of label propagation.
