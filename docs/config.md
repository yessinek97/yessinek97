## Configuration file

All experiment related parameters can be changed in a global configuration file. A default global
configuration file is available at `biondeep-ig/biondeep_ig/configuration/train.yml`.

All model related parameters can be changed in model configuration files. Defaults for each
available model are located at `biondeep-ig/biondeep_ig/configuration/model_config/<model>.yml`. The
model configuration file should be indicated under `models` in the global configuration file to run
the respective model per experiment.

In the following more details on the parameters are provided.

Multiple models can be defined in the global configuration file.

## Global variables: main configuration file

- neptune_logs :
  - Log to neptune
- experiments:
  - List of experiments. All experiments in the list will be run.
- label:
  - Column name of the target in the datasets
- feature_paths:
  - Paths to the feature lists to be evaluated. All feature lists in this list will be evaluated per
    experiment and model.
- eval_id_name:
  - Optional column name in the test dataset to get evaluation results grouped by this.
- observations_number:
  - Optional k for top-k metrics.
- comparison_score:
  - Column name in the datasets with scores of a comparison model.
- print_evals:
  - print model convergence.
- processing :

  - remove_proc_data :
    - If True the `data_proc` directory will be removed from the checkpoint when the train ends
  - trainable_features :
    - Name of the features dictionary file which contains the list of the `float` , `int` ,
      `categorical` features with the useful `ids` list and `id` column. The file must be saved
      under `configuration/features`
  - exclude_features :
    - List of features which will be excluded from the trainable features
  - validation_strategy:
    - If True use the available validation strategy splitter to generate the `split_column` if it's
      False use the provided `split_column` in the train data.
  - isunlabelled:
    - Placeholder for unlabelled dataset in the future.
  - fill_nan_method:
    - Automatically fill NaNs in the datasets can be one of `keep`, `mean`.
  - seed:
    - Global random seed
  - validation_ratio:
    - In case of SingleModel experiment determines the size of the validation set.
  - fold:
    - In case of KFold experiment determines how many folds will be created.
  - normalizer :

- models:
  - List of model configuration files located at
    `biondeep-ig/biondeep_ig/configuration/model_config/<model>.yml`. All models in the list will be
    run per experiment.
- evaluation :
  - threshold: for precision, recall and f1
  - metrics: Available metrics to test topk, roc, logloss, precision, recall, f1
  - metric_selector: What's the metric you can use for selecting best model it can be
    topk_20_patientid
  - metrics_selector_higher: It can be True or False
  - data_name_selector: It can be test or validation
  - prediction_name_selector: Generally `prediction_mean`
  - monitoring_metrics: metrics to monitor among these or all of them (roc, topk ,
    topk_20_patientid, topk_patientid)

## Global variables - Hyper-Parameters Tuning configuration file

The difference to the main config file is:

- Tune config file: tune_configuration

- Models: should be related to the tune model. for example use the config under
  /biondeep_ig/confirguration/tune_xgboost.yml

## Global variables - Modularization configuration file

The difference to the main config file is:

configs: list of configuration files that describe each separate model. Only Singleexperiment,
single model and single feature list is supported, i.e. one should input already optimized models
per module.
