## Configuration file

All experiment related parameters can be changed in a global configuration file. A default global
configuration file is available at `configuration/default_configuration.yml`.

In the following more details on the parameters are provided.

```yaml
neptune_logs: # Whether to log the run to Neptune or not.

experiments: # List of experiments. All experiments in the list will be run.

label: # Column name of the target in the datasets.

feature_paths: # Paths to the text files listing the features to be evaluated.
  # All feature lists in this list will be evaluated per experiment and model.

processing:
  process_data: #True/False if True allows the framework to process the data if the provided data is not processed if False the framework will not do any processing to the data,Default value is set to True
  remove_unnecessary_folders: # If True the `data_proc`, `features` and `Kfold experiments` directories will be removed from the checkpoint when the train ends
  trainable_features: # Name of the features dictionary file which contains the list of the `float` , `int` ,
    # `categorical` features with the useful `ids` list and `id` column. The file must be saved
    # under `configuration/features`
  exclude_features: # List of features indicators which will be excluded from the trainable features, in this list you can either use:
  # - prefixes: features that start with a specific prefix like gene_ontology/rna_localization/        go_cc_rna_loc/neofox
  # - patterns: features that include some pattern indicators like tested/nontested
  # - suffixes: features that end with a specific suffix like mhci or pmhc
  # - feature name: the whole feature name like rnalocalization_p_membrane
  #example:

    # exclude_features:
    #   - neofox
    #   - tested
    #   - pmhc
    #   - mhci
    #   - gene_ontology
    #   - rna_localization
    #   - go_cc_rna_loc
    #   - rnalocalization_p_membrane
  validation_strategy: # If True use the available validation strategy splitter to generate the `split_column` if it's
    # False use the provided `split_column` in the train data.
  isunlabelled: # Whether to use placeholder for unlabelled dataset in the future.
  fill_nan_method: # Method used to fill NaNs in the datasets. One of be one of `keep`, `mean`.
  seed: # Global random seed value
  validation_ratio: # Relative size of the validation set for SingleModel experiment.
  fold: # Number of folds used for the KFold experiment.
  normalizer: # Data normalization techniques used in the datasets.
  normalizer_per_id:  # Data normalization techniques applied by patient id in the datasets.
  normalizer_id: # patientid column.
  nan_ratio: #The maximum ratio of Nan value which the framework  could be accepted
  expression_name: # The name of the expression columns which the framework will use
  expression_column_name: #The final name of the expression column, the framework will rename `expression name` to `expression column_name` and use it, it's recommended to set expression_column_name to a name that exists in the trainable_features configuration  otherwise the expression will be taken into consi deration


models: # List of model configuration files located at
  # `configuration/model_config/<model>.yml`.
  # All models in the list will be run per experiment.

evaluation:
  print_evals: # Whether to print model convergence or not.
  comparison_score: # Column name in the datasets with scores of a comparison model.
  eval_id_name: # Optional column name in the test dataset to get evaluation results grouped by this.
  observations_number: # Optional k for top-k metrics.
  threshold: # Threshold for binary classification (used to compute precision, recall and f1).
  metrics: # Available metrics to test topk, roc, logloss, precision, recall, f1.
  metric_selector: # Metric used to select the best model (e.g. topk_20_patientid, topk, ...).
  metrics_selector_higher: # Whether higher values of `metrics_selector` means better model or not.
  data_name_selector: # One of `test` or `validation`.
  prediction_name_selector: # Generally `prediction_mean`.
  monitoring_metrics: # Metrics to monitor. One or many of: roc, topk, topk_20_patientid, topk_patientid
```


## Global variables - Hyper-Parameters Tuning configuration file

 #TBC
