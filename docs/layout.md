## Layout

Depending on the experiment type the framework will automatically create structured output of the
general form:

```bash
<>/biondeep-ig/biondeep_ig/models/<name>/.
├── <experiment_type>
│   ├── <feature_list>
│   │   └── <Model>
│   │       ├── checkpoint
│   │       │   ├── <split_0>
│   │       │   │   └── model.pkl
│   │       │   │   └── shap_plot.jpg
│   │       │   │   └── model.log
│   │       │   ├── split_1
│   │       │   │   └── model.pkl
│   │       │   ├── ...
│   │       ├── <configuration_file>
│   │       ├── eval
│   │       │   ├── test_metrics.yaml
│   │       │   ├── results.csv
│   │       │   ├── train_metrics.yaml
│   │       │   └── validation_metrics.yaml
│   │       │   └── curve_plots
│   │       │   ├── features_importances.jpg
│   │       ├── features.txt
│   │       └── prediction
│   │           ├── test.csv
│   │           └── train.csv
│   │           └── validation.csv

```

You can find more information on the results under the best experiment:

- prediction:

  - test.csv:
    - Copy of test dataset with model predictions appended.
  - train.csv:
    - Copy of train dataset with model predictions appended.
  - validation.csv:
    - Copy of validation dataset with model predictions appended.

- eval:

  - test_metrics.x:
    - Detailed summary of experiment/model performance on test dataset in terms of scalar metrics.
  - train_metrics.x:
    - Detailed summary of experiment/model performance on train dataset in terms of scalar metrics.
  - validation_metric:
    - Detailed summary of experiment/model performance on validation dataset in terms of scalar
      metrics.
  - curve plots:
  - Plots showing precision and recall for the tree dataset train, validation and test.
  - results: a file containing the results of the experiment.

- checkpoint:

  - model.pkl:
    - Pickle dump of the trained model.
  - model.log:

    - Model logging.

  - shap:

    - Shapley values of the best model showing the 20 most important features.

  - configuration.yml: a file containing the best combination of parameters and best feature list.

  - features.txt: the list of features from the best feature.

Additionally the framework logs intermediate results and progress to two files:

- biondeep-ig/biondeep_ig/models/<name>/<experiment_name>/<features_list>/<model_type>/runs.log:
  - progress and intermediate results of the current run
- biondeep-ig/biondeep_ig/models/<name>/<experiment_name>/<features_list>/<model_type>/models.log:
  - model convergence
