# Data configuration files

This page includes the description of multiple configuration files used in the **IG Framework** for **preprocessing** and **feature selection** purposes.

## Preprocessing

### Processing configuration file

This configuration file defines the settings for the main **processing** pipeline as part of data preprocessing. It can be found at `configuration/processing_configuration.yml`.

#### **PS**: Please note that all column names must be lowercase

Here is a detailed description of this configuration file:

```yml
data_version: # This argument specifies the version of the data.
data_type: # This argument defines the data type :Netmhcpan or BionNDeep, use it  with data_version to create the output directory with a name of "data_proc_{data_version}_{data_type}
push_gcp: #TODO # This boolean argument specifies whether to push the output data to gcp bucket or not (True or False).
processing: # The processing section lists the settings for the preprocessing pipeline.
  legend: # The legend section is used to filter features
    path : # This argument sets the legend file name where the file should be under the parent directory of the train data.
    filter_column: # This argument defines the column name used to filter features.
    value: # This argument defines the value inside the filter_column which represents the feature to take.
    feat_col_name: # This argument defines the column's name which hold the features name.
    remove_duplicated: # Remove duplicate rows based on specific columns
      remove: True # This argument defines whether to remove duplicate rows for train.
      train_columns: # This argument defines the column name used to filter duplicate rows for train.
        - id
      remove_test: False # This argument defines whether to remove duplicate rows for test.
      test_columns: # This argument defines the column name used to filter duplicate rows for test.
        - id
  filter_rows : # This section is used to remove rows base on a given column.
    filtre: # This argument defines whether to filter rows or not.
    filter_column : # This argument defines the name of column to filter.
    value : # This argument defines the value used to remove rows.
  remove_missing_value: # This section is used to remove rows with missing values.
    remove: # This argument defines whether to remove rows with missing values or not.
    columns: # This argument defines the name of the column to check for missing values.
      - col1
      - col2
  label: cd8_any # This argument sets the target label name.
  id: id # This argument defines the unique id.
  ids: # This section provides a list of the columns will be saved with the processed data.
    - id
    - patientid
    - wt_27mer
    - mut_27mer
    - tested_peptide
    - wildtype_peptide
    - author/source
    - genename
  proxy_model_columns: # This section details Binder/presentation model section parameters.
    proxy_m_peptide: tested_peptide_netmhcpan4.1 # This argument sets the name of the best binder mutated peptide column.
    proxy_wt_peptide: wildtype_peptide_netmhcpan4.1 # This argument sets the associated WT peptide for the best binder mutated peptide column.
    proxy_allele: allele_netmhcpan4.1 # This argument sets the best allele column name.
    scores: # This section defines the  list of the available  Binder/presentation  scores and the associated new name.
      # real_score_name : This argument sets a new_name with a prefix e.g. proxy_score
      tested_score_netmhcpan4.1: proxy_score_presentation
      tested_best_rank_netmhcpan4.1: proxy_score_rank_presentation
      tested_ba_score_netmhcpan4.1: proxy_score_score_binding
      tested_ba_rank_netmhcpan4.1: proxy_score_rank_binding
  expression: # This section defines the expression settings
    filter: # This section defines the list of the keyword to filter all the expression columns.
      - exp
      - tcg
      - tpm
      - gtex
    raw_name: expression_for_model #  This argument sets the expression's name to use.
    name: expression # This argument sets a new name for the raw_name.
  exclude_features: # This section provides the list of excluded features by keyword or regex format.
    - cd4
    - cd8
    - any
    - wt
    - wildtype
    - nontest
    - prime_rank
    - prime_score
    - length_tested_peptide
  include_features: # This section provides a list of included features by keyword or regex format.
    - proxy_score
    - rnalocalization
    - go_term_rna
    - go_term_.*_embed
    - pathway
    - deeptap
    - specificity_gtex
  keep_include_features: False # This boolean argument selects whether or not to force processing to keep features from include list in case they will be removed during processing.
  keep_same_type: False # This boolean argument selects whether or not to force processing to keep only features with the same types between datasets .
  nan_ratio: 0.6 # This argument defines a ratio on the missing values.
features: # This section holds features settings.
  file_name: features.yml # This argument sets the name of features configuration file.
  float: True # This argument defines whether to choose using float features type or not.
  include_features_float: True # This argument defines whether force processing to keep the intersection between float type and include features in case of float = False.
  int: True  # This argument defines whether choose to use int features type or not.
  include_features_int: True # This argument sets whether force processing to keep the intersection between int type and include features in case of int = False.
  categorical: False # This argument sets whether choose to use categorical features type or not.
  include_features_categorical: True # This argument chooses whether force processing to keep the intersection between categorical type and include features in case of categorical = False.
  bool: False # This argument sets whether choose to use bool features type or not.
  include_features_bool: True # This argument selects whether to force processing to keep the intersection between bool type and include features in case of bool = False.
split: # This section lists Cross-validation settings.
  split_path: # This argument sets the path of the split file which will be used instead of computed
  kfold: True # This argument selects whether to use Kfold split or not.
  kfold_column_name: fold # This argument specifies the column name which will hold the kfold splits if kfold is True.
  nfold: 5 # This argument defines the number of fold if kfold is True.
  train_val: True # This argument specifies whether to use train/validation split or not.
  train_val_name: validation  # This argument defines the column name which will hold the train/validation split if train_val is True.
  val_size: 0.1 # This argument defines the validation size.
  source_split: true # This argument specifies whether to use source split or not.
  source_split_name: genename # this argument defines the column name which will hold the source split if source_split is True.
```
