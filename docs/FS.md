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
featureselection -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
featureselection -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c train_with_fs.yml -n  test_FS
```

or in combination with training:

```
 train -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
 train -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c train_with_fs.yml -n  test_FS_train
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
