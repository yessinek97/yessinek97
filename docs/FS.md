## Feature selection (FS) methods

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

### Configuration

#### Global

The global configuration of feature selection process is given under the `FS` key of a training
configuration file (e.g. `configuration/train_with_fs.yml`).

The feature selection configuration contains the following:

```yaml
FS:
  n_feat: 20 # target number of selected features + number of forced features
  force_features: # features that will be forcefully selected
    - "tested_score_biondeep_mhci"
    - "expression"
    - "presentation_score"
  FS_methods: # list of configuration files used for feature selections
    - Fspca.yml
    - Fsxgboost.yml
    - Fsxgboostshap.yml
    - Fsrfgini.yml
    - Fscossim.yml
    - Fscorr.yml
    - Fssfmlr.yml
    - Fsrfelr.yml
    - Fssfmxgboost.yml
    - Fsrfexgboost.yml
    - Fsrelief.yml
    - Fsmi.yml
    - Fsboruta.yml
```

#### Per method

The configuration files for the every feature selection method (mentioned in the global
configuration file) are located in:

```bash
configuration/FS_config/FS_method_config/<method>.yml
```

### Execution

To activate feature selection the respective set of parameters have to be defined in the global
configuration file - see below. We can either run feature selection alone:

```bash
featureselection -train <train_data> -c <configuration_file> -n <name>
```

Example:

```bash
featureselection -train data/public_2022_03_10\:clean\:latest.csv -c train_with_fs.yml -n  test_FS
```

The feature selection and training steps can also be called successively with a single command:

```bash
train -train <train_data>  -test <test_data>  -c <configuration_file> -n  <run_name>
```

Example:

```bash
train -train data/public_2022_03_10\:clean\:latest.csv -test optima_2022_03_10\:latest\:floats.csv -c train_with_fs.yml -n  test_FS_train
```

### Results

An analysis of each feature selection method is stored in:

```bash
configuration/FS_config/FS_feature_lists_created/<method>.pdf
```

The selected features lists for each method will be stored in:

```bash
models/<run_name>/features/<FS_method_name>.txt
```
