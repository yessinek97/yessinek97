## Requirements

- Conda to setup the environment

## Conda environment creation

```
# Create a conda env
conda env create -f environment.ig.train.yaml && conda activate biondeep_ig_train

# Install pre-commit hooks (optional for developing)
pre-commit install -t pre-commit -t commit-msg
```

## Data used

- Training/Validation dataset - Public data:
  - `gs://biondeep-data/optima/Datasrets_11_10_2021/publicIEDBFilteredTransformerS128_20210827_out_nancleaned_RemovedTestSets.tsv`
- Test dataset - Optima:
  - `gs://biondeep-data/optima/Datasrets_11_10_2021/optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv`
- Test dataset - Parkhust:
  - To be delivered
- Test dataset - Sahin et al.
  - `gs://biondeep-data/optima/Datasrets_11_10_2021/Sahin2017_TransformerS128_20210818_out.tsv`

## Train model

Training of a model can be launched with the `train` command using the provided configuration file,
the data file and a name. The parent directory from which the framework should be started is as
usual the /biondeep folder (not /biondeep/biondeep_ig).

```
python -m  biondeep_ig.main train -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
python -m  biondeep_ig.main train -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c train_configuration.yml -n  test_train
```

```
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

The trained models will be available in `biondeep_ig/models/<name>` directory

## Output

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
│   │       │   │   └── shape_plot.jpg
│   │       │   │   └── model.log
│   │       │   ├── split_1
│   │       │   │   └── model.pkl
│   │       │   ├── ...
│   │       ├── <configuration_file>
│   │       ├── eval
│   │       │   ├── ScoreSummary.pdf
│   │       │   ├── test_metrics.yaml
│   │       │   ├── test_results.csv
│   │       │   ├── train_metrics.yaml
│   │       │   └── train_results.csv
│   │       │   ├── features_importances.jpg
│   │       ├── features.txt
│   │       └── prediction
│   │           ├── test.csv
│   │           └── train.csv
```

- prediction/ScoreSummary.pdf:
  - Visual summary of the experiment performance. Columns: 1. Performance on validation (V), 2.
    Performance on test (SV). Rows: 1. Top-k metric, 2. AUC-ROC metric.
- prediction/test.csv:
  - Copy of test dataset with model predictions appended.
- prediction/train.csv:
  - Copy of train dataset with model predictions appended.
- eval/test_metrics.x:
  - Detailed summary of experiment/model performance on test dataset in terms of scalar metrics.
- eval/train_metrics.x:
  - Detailed summary of experiment/model performance on train dataset in terms of scalar metrics.
- model.pkl:
  - Pickle dump of the trained model.

Additionally the framework logs intermediate results and progress to two files:

- biondeep-ig/biondeep_ig/models/<name>/runs.log:
  - progress and intermediate results of the current run
- biondeep-ig/biondeep_ig/models/<name>/models.log:
  - model convergence

## Configuration file

All experiment related parameters can be changed in a global configuration file. A default global
configuration file is available at `biondeep-ig/biondeep_ig/configuration/final_configuration.yml`.

All model related parameters can be changed in model configuration files. Defaults for each
available model are located at `biondeep-ig/biondeep_ig/configuration/model_config/<model>.yml`. The
model configuration file should be indicated under `models` in the global configuration file to run
the respective model per experiment.

In the following more details on the parameters are provided.

## Available experiments

- SingleModel:
  - Single model training
  - Parameters:
    - validation_column: Column name of a binary feature indicating which samples should be used as
      validation. Alternatively `validation` if processing is active and takes care of it based on
      `processing:validation_ratio`.
    - plot_shap_values : plot shape values or not
- KfoldExperiment:
  - Classic Kfold experiment
  - Parameters:
    - split_column: Column name of a feature consisting of integers \[0,k\] indicating which samples
      should be used for which fold. Alternatively `fold` if processing is active and takes care of
      it based on `processing:fold`.
    - plot_shap_values: plot shape values or not
    - plot_kfold_shap_values: plot shape values for Kfold method or not
- DoubleKfold:
  - K times classic Kfold experiment
  - Parameters:
    - split_column: Column name of a feature consisting of integers \[0,k\] indicating which samples
      should be used for which fold. Alternatively `fold` if processing is active and takes care of
      it based on `processing:fold`.
- Tuning:
  - Hyperparameter optimization

Multiple experiments can be defined in the global configuration file.

## Available models

- XGBoost
- CatBoost
- Lgbm
- Logistic Regression

Multiple models can be defined in the global configuration file.

## Global variables - main configuration file

- neptune_logs :
  - Log to neptune
- experiments:
  - List of experiments. All experiments in the list will be run.
- label:
  - Column name of the target in the datasets
- benchmark_column:
  - Names of scores to decide on the best model in the final prints. Can be one of
    `prediction_average` ,`mean_topk` ,`max_topk` ,`mintopk`
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
- processing
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
- models:
  - List of model configuration files located at
    `biondeep-ig/biondeep_ig/configuration/model_config/<model>.yml`. All models in the list will be
    run per experiment.

## Train seed fold

The goal is to train the model with a list of number of folds and a list of seeds.

```
python -m  biondeep_ig.main train-seed-fold -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
python -m  biondeep_ig.main train-seed-fold -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c final_configuration_seed_fold.yml -n  test_seed_folds
```

```
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

### Global variables - train seed fold configuration file

The difference with the main config file is:

- seeds:
  - a list of random seed
- folds:
  - a list of of number of folds to be created. In case of single KFold experiment determines how
    many folds will be created.

## Hyper-Parameters Tuning

The goal is to find the best hyperparameters for a specific model, for example Xgboost. You can use
the resulted params with the training command.

```
python -m  biondeep_ig.main tune -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
python -m  biondeep_ig.main tune -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c tune_configuration.yml -n  test_tune
```

```
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

### Global variables - Hyper-Parameters Tuning configuration file

The difference to the main config file is:

- Tune config file: tune_configuration

- Models: should be related to the tune model. for example use the config under
  /biondeep_ig/confirguration/tune_xgboost.yml

## Modularization

The goal is to train multiple models each with a separate set of features, i.e. feature contraction.
Finally, a linear model will be trained on the predictions of each separate model to predict
immunogenicity. An example of a config file for this model type is given in
configuration/final_configuration_modular_train.yml

```
python -m  biondeep_ig.main modulartrain -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
python -m  biondeep_ig.main modulartrain -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c modular_train.yml -n  test_modulartrain
```

### Global variables - Modularization configuration file

The difference to the main config file is:

configs: list of configuration files that describe each separate model. Only Singleexperiment,
single model and single feature list is supported, i.e. one should input already optimized models
per module.

## Inference on a single model with a bunch of datasets

```
python -m  biondeep_ig.main Inference -c  <configuration_file>
```

```
Options:
  -c       TEXT    Configuration file to use [required]
```

## Eval trained models on a seperate dataset

You can launch a trainon with a single test or multiple test sets.

```
python -m  biondeep_ig.main eval -test  <data_Test_paths>  -n <models_folder_name>

python -m biondeep_ig.main eval -n a_trained_model -test   test1.tsv -test test2.tsv
```

```
Options:
  -Test       TEXT    Path to dataset [required]
  - n         TEXT    path tomodel folder name [required]
```

## Feature selection (FS)

Multiple automatic feature selection algorithms are implemented:

- XGBoost importance
- Relief
- Gini (Random forest)
- Boruta
- Correlation with target
- Cosine similarity with target
- Mutual information
- PCA
- Random feature elimination (XGBoost and LR)
- Select from model (XGBoost and LR)
- Shap values (XGBoost)

All config files for the respective feature selection method are located in:

configuration/FS_config/FS_method_config

To activate feature selection the respective set of parameters have to be defined in the global
configuration file - see below. We can either run feature selection alone:

```
python -m  biondeep_ig.main featureselection -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
python -m  biondeep_ig.main featureselection -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c FS_configuration.yml -n  test_FS
```

or in combination with training:

```
python -m  biondeep_ig.main train -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
python -m  biondeep_ig.main train -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c FS_configuration.yml -n  test_FS_train
```

the resulting feature lists will be stored in:

configuration/features/<target>/<FS_Alg_Name>.yml

### Global variables - FS configuration file

- FS: Feature selection key
- min_nonNaNValues: Minimum ratio of non-NaN values existing in a feature. If more NaN values exist
  feature will not be considered for FS.
- n_feat: Number of features to be selected with decreasing importance
- min_unique_values: Minimum number of unique values a feature has to have to be considered
- max_unique_values: Maximum number of unique values a feature has to have to be considered
- force_features: List of features that will be included in the feature list by force (Note:
  Currently n_feat + len(force_features))
- FS_methods: List of FS methods to execute

- IMPORTANT: There is a black list for features in: configuration/FS_config/FeatureExclude.yml which
  is automatically checked and respective features are not taken into consideration for FS.
