## Available experiments

### SingleModel:

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
    - operations: It can be the mean, max, min, median of the prediction folds.
    - statistics: of the trained models in terms of mean, max, and min of the TopK.

### DoubleKfold:

- K times classic Kfold experiment
- Parameters:
  - split_column: Column name of a feature consisting of integers \[0,k\] indicating which samples
    should be used for which fold. Alternatively `fold` if processing is active and takes care of it
    based on `processing:fold`.
    - operations: It can be the mean, max, min, median of the prediction folds.
    - statistics: of the trained models in terms of mean, max, and min of the TopK.

### Tuning:

- Hyperparameter optimization

Multiple experiments can be defined in the global configuration file.

## Available models

- `XgboostModel`
- `LgbmModel`
- `CatBoostModel`
- `LogisticRegressionModel`
- `LabelPropagationModel`

## Train seed fold

The goal is to train the model with a list of number of folds and a list of seeds.

```
train-seed-fold -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
train-seed-fold -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c train_seed_fold.yml -n  test_seed_folds
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
tune -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
tune -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c tune.yml -n  test_tune
```

```
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

## Modularization

The goal is to train multiple models each with a separate set of features, i.e. feature contraction.
Finally, a linear model will be trained on the predictions of each separate model to predict
immunogenicity. An example of a config file for this model type is given in
configuration/final_configuration_modular_train.yml

```
modulartrain -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
python -m  biondeep_ig.main modulartrain -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c modular_train.yml -n  test_modulartrain
```
