# Training-and-experimentation-configuration-files

This page includes the description of multiple configuration files used in the **IG Framework** for **training and experimentation** purposes.

## Training and experimentation
### Default configuration file

The **default configuration file** is used to define the skeleton of the **study** we want to conduct. All of the study settings can be defined in a global configuration filethat can be found at `configuration/default_configuration.yml`.

This is a detailed description of a **default configuration file** main sections and arguments:

```yaml
neptune_logs: # This argument is used to control whether we want log the ablation using Neptune or not.

experiments: # This section includes the list of experiments (SingleModel, KfoldExperiment, DoubleKfold, KfoldMultiSeedExperiment) that will be included in the ablation.

label: # This argument specifies the target column name for the input dataset.

feature_paths: # This section includes path to separate text files listing the features to be evaluated.
  # All feature lists in this section will be evaluated per experiment and model.

processing: # This section specifies the processing settings

  process_data: # This argument is a boolean that controls whether to process the input dataset or not (Default value is set to True).

  remove_unnecessary_folders: # This argument is a boolean that controls whether to remove the unnecessary folders (data_proc, features and experiments folders) after training or not(Default value is set to False).

  trainable_features: # This argument specifies the name features dictionary file which contains the list of the `float` , `int` ,
    # `categorical` features with the useful `ids` list and `id` column. The file must be saved
    # under at the same directory with the data files (train, test).
  exclude_features: # This argument specifies the list of features indicators which will be excluded from the trainable features, in this list you can either use:

    # - prefixes: features that start with a specific prefix like gene_ontology/rna_localization/        go_cc_rna_loc/neofox
    # - patterns: features that include some pattern indicators like tested/nontested
    # - suffixes: features that end with a specific suffix like mhci or pmhc
    # - feature name: the whole feature name like rnalocalization_p_membrane

  # Example:

    exclude_features:
      - neofox
      - tested
      - pmhc
      - mhci
      - gene_ontology
      - rna_localization
      - go_cc_rna_loc
      - rnalocalization_p_membrane

  validation_strategy: # This argument is a boolean that selects whether to use the available validation strategy splitter to generate the `split_column` if True or use the provided `split_column` in the train data if False.

  isunlabelled: # This argument defines whether to use placeholder for unlabelled dataset in the future or not.

  fill_nan_method: # This argument specifies the method to use to fill the missing values in the datasets(mean, median,keep).

  seed: # This argument defines the global random seed value.

  validation_ratio: # This argument specifies the relative size of the validation set for SingleModel experiment.

  fold: # This argument specifies the number of folds used for the KFold experiment.

  normalizer: #TODO: This argument specifies the used normalizer for the input dataset.

  nan_ratio: #This argument specifies the maximum ratio of missing values that the framework can tolerate.

  expression_name: # This argument specifies the name of the expression columns which the framework will use.

  expression_column_name: #This arguments defines the final name of the expression column, the framework will rename `expression name` to `expression column_name` and use it, it's recommended to set expression_column_name to a name that exists in the trainable_features configuration otherwise the expression will be taken into consideration.


models: # This section defines a list of model configuration files located at
  # `configuration/model_config/<model>.yml`.
  # All models in the list will be run for each experiment.

evaluation: # This section defines the evaluation settings for our ablation.

  print_evals: # This argument specifies Whether to print model convergence or not.

  comparison_score: # This argument specifies the column name in the datasets with scores of a comparison model.

  eval_id_name: # This argument specifies the optional column name in the test dataset used to group the evaluation results.

  observations_number: # This argument defines the optional k number for top-k metrics per eval_id evaluation.

  threshold: # This argument defines the threshold for binary classification (used to compute precision, recall and f1).

  metrics: # This section lists the available metrics to test topk, roc, logloss, precision, recall, f1.

  metric_selector: # This argument defines the metric used to select the best model (e.g. topk_20_patientid, topk, ...).

  metrics_selector_higher: # This boolean argument specifies Whether the highest values of `metrics_selector` means better model or not.

  data_name_selector: # This argument defines the data split used to pick the best model (`test` or `validation`).

  prediction_name_selector: # This argument specifies the prediction name selector (Generally `prediction_mean`).
  monitoring_metrics: # This section lists the metrics to monitor. One or many of: roc, topk, topk_20_patientid, topk_patientid

FS: This section defines the feature selection settings.

  n_feat: # This argument specifies the final number of features to select.
  keep_features_number: # This boolean is used to force the number selected features to be equal to n_feat if it's True.
  force_features: # This section defines features that will be forcefully selected
  # Example:

    - "tested_score_biondeep_mhci"
    - "expression"
    - "presentation_score"

  n_thread: This argument specifies the number of parallel threads used by the Feature selection method.
  FS_methods: This section lists the configuration files of feature selection methods that can be found in `configuration/FS_config`.
```

### Train configuration file

This configuration file is the same one as the **Default configuration file**. It includes settings for (netptune_logs, experiments, label, processing, models, evaluation and FS) an it's located at `configuration/train_configuration.yml`. Here is an example of a **training configuration file**:

#### **Example**

```yaml
neptune_logs: False
experiments:
  KfoldExperiment:
    split_column: fold
    plot_shap_values: True
    plot_kfold_shap_values: True # False when using Logistic regression
    operation:
      - mean
    statics:
      - mean
  KfoldMultiSeedExperiment:
    split_column: fold
    plot_shap_values: True
    plot_kfold_shap_values: True # False when using Logistic regression
    model_seed: false
    split_seed: true
    nbr_seeds: 10
    print_seeds_evals: False
    operation:
      - mean
    statics:
      - mean
label: cd8_any
processing:
  process_data: False
  remove_unnecessary_folders: false
  trainable_features: "features.yml"
  exclude_features: # You can use prefixes(neofox..),suffixes(mhci,pmhc) patterns(tested) or the full feature name
    - neofox

  validation_strategy: false
  isunlabelled: false
  #normalizer:
  fill_nan_method: keep
  seed: 1994
  validation_ratio: 0.1
  fold: 5
  nan_ratio: 0.6
  expression_name: expression_for_model
  expression_column_name: expression
models:
  - xgboost_config.yml

evaluation:
  print_evals: True
  comparison_score: tested_pres_score_biondeep_mhci
  eval_id_name: patientid
  observations_number: 20
  threshold: 0.3
  metrics:
    - topk
    - roc
    - logloss
    - precision
    - recall
    - f1
  metric_selector: topk
  metrics_selector_higher: True
  data_name_selector: validation # test , validation
  prediction_name_selector: "prediction_mean"
  monitoring_metrics:
    - roc
    - topk
    - topk_20_patientid
    - topk_patientid
FS:
  n_feat: 20
  keep_features_number: False
  separate_forced_features: False
  force_features:
    - expression
  n_thread: 10
  FS_methods:
    - Fspca.yml #  Here we used PCA for feature selection
    - Fsxgboost.yml #  Here we used Xgboost for feature selection

```

### Train seed_fold configuration file

This configuration file is the same as **Train configuration file** but with updated seeds and folds. It can be found at `configuration/train_seed_fold.yml`. Here is an example of Train seed_fold configuration file:

```yaml
neptune_logs: False
experiments:
  KfoldExperiment:
    split_column: fold
    plot_shap_values: True
    plot_kfold_shap_values: True # False when using Logistic regression
    operation:
      - mean
      - max
      - min
      - median
    statics:
      - mean
      - max
      - min
  SingleModel:
    validation_column: validation
    plot_shap_values: True
label: cd8_any
feature_paths:
  - train_features
processing:
  process_data: False
  remove_unnecessary_folders: False
  trainable_features: features_float.biondeep_02_06_2022_transformer_public_v2
  # exclude_features:
  validation_strategy: true
  isunlabelled: false
  #normalizer:
  fill_nan_method: keep
  seeds: [1994, 2002] # Here we used multiple seeds
  validation_ratio: 0.1 # Here we used multiple folds
  folds: [5, 4, 6]
  nan_ratio: 0.6
  expression_name: geneexpression_final
  expression_column_name: expression

models:
  - xgboost_config.yml
evaluation:
  print_evals: True
  comparison_score: tested_score_biondeep_mhci
  observations_number: 20
  threshold: 0.3
  metrics:
    - topk
    - roc
    - logloss
    - precision
    - recall
    - f1
  metric_selector: topk
  metrics_selector_higher: True
  data_name_selector: test # test , validation
  prediction_name_selector: "prediction_mean"
  monitoring_metrics:
    - roc
    - topk

FS:
  n_feat: 20
  keep_features_number: False
  separate_forced_features: False
  force_features:
    - expression
  n_thread: 10


```

### Multi-train configuration file

This configuration file defines the setting of an **ablation study** having multiple runs.
This file can be found at `configuration/multi_train_configuration.yml`.
Here is a detailed description of the configuration file settings:

```yaml
experiments: # This section includes the list of the runs we want to launch during the ablation.

  experiment1: # This section defines the settings of experiment1.
    dataset: # This subsection is optional, it specifies whether we're going to use a specific dataset for this run.
      train: # This argument specifies the training file of the used data
      test: # This argument specifies the test file of the used data
    processing: # This section specifies the processing settings for experiment1.
      exclude_features: # This argument specifies the list of features indicators which will be excluded from the trainable features, in this list you can either use:

      # - prefixes: features that start with a specific prefix like gene_ontology/rna_localization/        go_cc_rna_loc/neofox
      # - patterns: features that include some pattern indicators like tested/nontested
      # - suffixes: features that end with a specific suffix like mhci or pmhc
      # - feature name: the whole feature name like rnalocalization_p_membrane

    # Example:

      exclude_features:
        - tested
        - pmhc
        - gene_ontology
        - go_cc_rna_loc
    FS: # This section includes the feature selection settings of experiment1.
      force_features: # This section defines features that will be forcefully selected
      # Example:

        - "tested_score_biondeep_mhci"
        - "expression"
        - "presentation_score"
    experiment2: # This section includes the settings for experiment2.
      processing: # This section specifies the processing settings for experiment1.
        exclude_features: # This argument specifies the list of features indicators which will be excluded from the trainable features, in this list you can either use:

        # - prefixes: features that start with a specific prefix like gene_ontology/rna_localization/        go_cc_rna_loc/neofox
        # - patterns: features that include some pattern indicators like tested/nontested
        # - suffixes: features that end with a specific suffix like mhci or pmhc
        # - feature name: the whole feature name like rnalocalization_p_membrane

      # Example:

        exclude_features:
          - neofox
          - rna_localization


      FS: # This section includes the feature selection settings of experiment1.
        force_features: # This section defines features that will be forcefully selected
      # Example:

          - "presentation_score"

params: # This section includes the additional settings of the multi-train ablation.
  metrics: # This section lists the used metrics to evaluate the multi-train ablation.

  # Example:

    metrics:
      - topk
      - topk_20_patientid
```

### Tuning configuration file

This configuration file is used to define the needed settings for **Hyper-parameter tuning**. It can be found at `configuration/tune_configuration.yml`. Here is an example detailing this file's parameters:

```yaml
neptune_logs: False

experiments:
  SingleModel:
    validation_column: validation
    plot_shap_values: False
  KfoldExperiment:
    split_column: fold
    plot_shap_values: True
    plot_kfold_shap_values: True
    operation:
      - mean
      - max
      - min
      - median
    statics:
      - mean
      - max
      - min
label: cd8_any
feature_paths:
  - train_features
tuning: # This section is used to define the settings for tuning the model's hyper  parameters
  maximize: True # This argument chooses whether to maximize the scores.
  nbr_trials: 1 # This argument defines the number of trials for tuning.
processing:
  process_data: False
  trainable_features: "features_float.biondeep_02_06_2022_transformer_public_v2"
  validation_strategy: true
  isunlabelled: false
  fill_nan_method: keep
  seed: 20015
  validation_ratio: 0.1
  fold: 4
models:
  - tune_xgboost.yml
evaluation:
  print_evals: False
```

## Quickstart experiments
### Quickstart configuration file

The Quickstart configuration file uses the same schema, as the **Default configuration file** with a few changes. It can be found at `configuration/quickstart_{operation}.yml` with operations that can be either **train**, **seed_fold** or **tune**.

#### **Quickstart Train configuration file**

This file is used to run a dummy experiment to test the framework main functionalities. It can be found at  Here is an example of what could the configuration file be:
```yaml
neptune_logs: False
experiments:
  KfoldExperiment:
    split_column: fold
    plot_shap_values: True
    plot_kfold_shap_values: True
    operation:
      - mean
      - max
      - min
      - median
    statics:
      - mean
      - max
      - min
  KfoldMultiSeedExperiment:
    split_column: fold
    plot_shap_values: True
    plot_kfold_shap_values: True
    model_seed: false
    split_seed: true
    nbr_seeds: 2 # We fixed the number of seeds to 2.
    print_seeds_evals: false
    operation:
      - mean
    statics:
      - mean
label: cd8_any
feature_paths:
  - train_features
processing:
  process_data: True
  remove_unnecessary_folders: False
  trainable_features: "features_quickstart.yml" # Here we used the quickstart features instead of all the features.

  validation_strategy: True
  isunlabelled: false
  fill_nan_method: keep
  seed: 1994
  validation_ratio: 0.1
  fold: 5
  expression_name: expression
  expression_column_name: expression
models:
  - xgboost_config.yml # We only used the XGBoost model.

evaluation:
  print_evals: True
  comparison_score: tested_score_biondeep_mhci
  eval_id_name: patientid
  observations_number: 20
  threshold: 0.3
  metrics:
    - topk
    - roc
    - logloss
    - precision
    - recall
    - f1
  metric_selector: topk
  metrics_selector_higher: True
  data_name_selector: test # test , validation
  prediction_name_selector: "prediction_mean"
  monitoring_metrics:
    - roc
    - topk
    - topk_20_patientid
    - topk_patientid
FS:
  n_feat: 20
  n_thread: 5
  force_features:
    - tested_score_biondeep_mhci
    - expression
    - tested_presentation_biondeep_mhci
  FS_methods:
    - Fspca.yml # We only used PCA for feature selection.
```
#### **Quickstart seed_fold configuration file**

This file is used to run a dummy experiment to test the framework main functionalities but with multiple seeds and multiple folds as shown in the example below:
```yaml
neptune_logs: False
experiments:
  KfoldExperiment:
    split_column: fold
    plot_shap_values: False
    plot_kfold_shap_values: False
    operation:
      - mean
      - max
      - min
      - median
    statics:
      - mean
      - max
      - min
  SingleModel:
    validation_column: validation
    plot_shap_values: True
label: cd8_any
feature_paths:
  - train_features
processing:
  process_data: true
  remove_unnecessary_folders: False
  trainable_features: features_quickstart.yml
  validation_strategy: true
  isunlabelled: false
  fill_nan_method: keep
  seeds: [1994, 2002] # Here we run those experiments with multiple seeds.
  validation_ratio: 0.1
  folds: [5, 4, 6] # Here we run those experiments with multiple folds.
  nan_ratio: 0.6
  expression_name: geneexpression_final
  expression_column_name: expression
  # keep_best_exp_only: True

models:
  - xgboost_config.yml
evaluation:
  print_evals: True
  comparison_score: tested_score_biondeep_mhci
  observations_number: 20
  threshold: 0.3
  metrics:
    - topk
    - roc
    - logloss
    - precision
    - recall
    - f1
  metric_selector: topk
  metrics_selector_higher: True
  data_name_selector: test # test , validation
  prediction_name_selector: "prediction_mean"
  monitoring_metrics:
    - roc
    - topk
```
#### **Quickstart tune configuration file**

This file is used to tune an existing experiment with multiple tuning settings as shown in the example below:
```yaml
neptune_logs: False

experiments:
  SingleModel:
    validation_column: validation
    plot_shap_values: False
  KfoldExperiment:
    split_column: fold
    plot_shap_values: True
    plot_kfold_shap_values: True
    operation:
      - mean
      - max
      - min
      - median
    statics:
      - mean
      - max
      - min
label: cd8_any
feature_paths:
  - train_features
tuning: # This section defines the tuning settings.
  maximize: True # This argument chooses whether to maximize the scores.
  nbr_trials: 1 # This argument defines the number of trials for tuning.
processing:
  process_data: false
  trainable_features: features_quickstart.yml
  validation_strategy: true
  isunlabelled: false
  fill_nan_method: keep
  seed: 20015
  validation_ratio: 0.1
  fold: 4
models:
  - tune_xgboost.yml
evaluation:
  print_evals: False
  ```

## CIMT model experiment

### CIMT model configuration file

This configuration file belongs to the **CIMT model** conducted ablation. It includes difference configuration files respectively for **feature selection** and **training** and a **general** configuration file linking them together.

####  **CIMT General configuration file**

Here are some details about the provided settings:

```yaml
General:
  features_selection: # features selection configuration
    n_splits:  # number os splits
    seed:  # seed which will be used to split the train data
    Ntop_features: # number of the final selected  features
    default_configuration: # default configuration for features selection process
  train: # train configuration file
    n_splits: # number os splits
    seed: # seed which will be used to split the train data
    default_configuration: # default configuration for training  process

experiments:
  experiment_name_1: # experiment name
    features_selection :
      configuration:
        section_to_change : #the section name which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
          parameter_1 :  #section's parameter which will be modified
          parameter_2 :
        section_to_change : #the section name  which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
          parameter_1 :  #section's parameter which will be modified
          parameter_2 :
    train :
      configuration:
        section_to_change : #the section name which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
          parameter_1 :  #section's parameter which will be modified
          parameter_2 :
        section_to_change : #the section name  which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
          parameter_1 :  #section's parameter which will be modified
          parameter_2 :
  experiment_name_2:
    features_selection :
      configuration:
        section_to_change : #the section name which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
          parameter_1 :  #section's parameter which will be modified
          parameter_2 :
        section_to_change : #the section name  which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
          parameter_1 :  #section's parameter which will be modified
          parameter_2 :

  experiment_name_3:
  # if no changed the experiment will use the default parameters defined in the default configuration fils
```
Regarding the `configuration` sections for both `features_selection` and `train` are the normal configuration files used with the `train` command
predefined configuration files are implemented under the configuration file
- `CIMT.yml`: main configuration file
- `CIMTFeaturesSelection.yml`: train configuration file for the features selection process
- `CIMTTraining.yml`: train configuration file for the training  process
