## Layout

Depending on the experiment type the framework will automatically create structured output of the
general form:

```
models/<name>
      ├── <experiment_1>
      │   ├── <features_set_1>   # One for each feature selection and feature list defined in text file.
      │   │   └── <model>
      │   │       ├── checkpoint  # Pickle files and training iteration logs of the model.
      │   │       ├── configuration.yml  # Subset of the global configuration file with only parameters of the best experiment.
      │   │       ├── eval
      │   │       │   ├── ScoreSummary.png  # Global TopK and ROC scores on train/validation/test splits.
      │   │       │   ├── curve_plots  # Precision-recall plots on the train/validation/test splits.
      │   │       │   ├── results.csv  # Performance of experiment/model train/validation/test splits.
      │   │       │   ├── shap.jpg  # Shapley values of the best model showing the 20 most important features.
      │   │       │   ├── test_metrics.yaml  # Performance of experiment/model on test dataset in terms of scalar metrics.
      │   │       │   ├── train_metrics.yaml  # Performance of experiment/model on train dataset in terms of scalar metrics.
      │   │       │   └── validation_metrics.yaml  # Performance of experiment/model on validation dataset in terms of scalar.
      │   │       ├── features.txt  # features list.
      │   │       └── prediction
      │   │           ├── test.csv  # Copy of test dataset with model predictions appended.
      │   │           └── train.csv # Copy of train dataset with model predictions appended.
      │   │           └── validation.csv # Copy of validation dataset with model predictions appended.
      │   └── <features_set_2>
      │   └── ...
      ├── <experiment_2>
      ├── ...
      ├── FS_configuration.yml  # Feature selection parameters (only if FS is run).
      ├── InfoRun.log  # Progress and intermediate results of the current run.
      ├── best_experiment  # Experiment/model with best performances.
      ├── comparison_score  # Metrics computed on the basis of `evaluation:comparison_score` feature.
      ├── configuration.yml  # Copy of the global configuration file passed to the command line.
      ├── data_proc
      │   ├── featureizer.pkl  # Pickle file of the featurizer object used to preprocess the data.
      │   ├── features_configuration.yml  # Features classification by data type.
      │   ├── <train>  # Copy of the train dataset
      │   ├── <test>  # Copy of the test dataset
      ├── features  # Text files containing the list of selected features.
      └── results.csv  # Best scores of all experiment/model.
```

The `best_experiment` directory stores the trained model files, inference and metrics which has the
highest score defined in the [global configuration file](./config.md#Global-configuration-file):

```yaml
evaluation:
  metric_selector:
  metrics_selector_higher:
```
