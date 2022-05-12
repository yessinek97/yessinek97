Multiple experiments can be defined in the
[global configuration file](./config.md#Global-configuration-file) as such:

```yaml
experiments:
  Experiment1:
    param1_1: value1_1
    param1_2: value1_2
  Experiment2:
    param2_1: value2_1
    param2_2: value2_2
```

The available experiments as listed below.

## SingleModel

This experiment trains a single model. The following parameters can be set:

```yaml
experiments:
  SingleModel:
    validation_column:# Column name of a binary feature indicating
      # which samples should be used as validation.
      # Alternatively `validation` if processing is active
      # and takes care of it based
      # based on `processing:validation_ratio`.
    plot_shap_values: # Whether to plot shape values or not.
```

## KfoldExperiment

This experiment performs the K-fold cross validation training strategy. THe following parameters can
be set:

```yaml
experiments:
  KfoldExperiment:
    split_column:# Column name of a feature consisting of integers \[0,k\]
      # indicating which samples should be used for which fold.
      # Alternatively `fold` if processing is active
      # and takes care of it based on `processing:fold`.
    plot_shap_values: # Whether to plot shape values or not.
    plot_kfold_shap_values: # Whether plot shape values for Kfold method or not.
    operations: # List of operation on the prediction folds: mean, max, min, median.
    statistics:# List of statistics of the trained models performance
      # on the TopK score: mean, max, and min.
```

## DoubleKfold

This experiment performs `two-times K-fold` cross validation training strategy. THe following
parameters can be set:

```yaml
experiments:
  DoubleKfold:
    split_column:# Column name of a feature consisting of integers \[0,k\]
      # indicating which samples should be used for which fold.
      # Alternatively `fold` if processing is active
      # and takes care of it based on `processing:fold`.
    plot_shap_values: # Whether to plot shape values or not.
    plot_kfold_shap_values: # Whether plot shape values for Kfold method or not.
    operations: # List of operation on the prediction folds: mean, max, min, median.
    statistics:# List of statistics of the trained models performance
      # on the TopK score: mean, max, and min.
```
