# Data configuration files

This page includes the description of multiple configuration files used in the **IG Framework** for **feature generation**, **preprocessing** and **feature selection** purposes.

## Feature generation
## Gene ontology configuration file

This configuration file is used by the **Gene Ontology** pipeline to specify the settings for feature generation. It can be found at `configuration/gene_ontology.yml`. Here are some details about this configuration file:

```yaml
# Base datasets
dataset: # This section lists the used datasets to generate the needed features.
  dataset1: # This section describes dataset1.
    version: # This argument is used to specify the dataset version.
    # Example:
    version: "16_11_2022"
    paths: # This section lists the multiple data files included in dataset1.
      file1: # File1 path (It can be either local or GS path).
      file2: # File2 path (It can be either local or GS path).
    processing:
      trainable_features: # dataset1 features config file name
  dataset2: # This section describes dataset2.
    version: # This argument is used to specify the dataset version.
    # Example:
    version: "24_10_2022"
    paths: # This section lists the multiple data files included in dataset1.
      file1: # File1 path (It can be either local or GS path).
      file2: # File2 path (It can be either local or GS path).
    processing:
      trainable_features: # dataset2 features config file name
go_features: # This section describes the Ready-to-use gene ontology embeddings (cc,bp,mf)

  version: "16_11_2022" # This argument is used to specify the embeddings version.
  embedding_paths: # This section specifies the embedding paths (local, GS path)
    file1: # File1 embedding path (It can be either local or GS path).
    file2: # File2 embedding path (It can be either local or GS path).
    file3: # File3 embedding path (It can be either local or GS path).
    file4: # File4 embedding path (It can be either local or GS path).


  dimensionality_reduction: # This section describes the Dimensionality reduction settings to reduce embedding vectors.
    technique: # This argument defines the dimensionality reduction technique (pca,lda,lsa,tsne).
    n_components: # This argument specifies the number of components used to reduce the embeddings vectors shape.
  save_embeddings: # This boolean defines whether to save the embeddings independently or not.
```

## Preprocessing
### Processing configuration file

This configuration file defines the settings for the main **processing** pipeline as part of data preprocessing. It can be found at `configuration/processing_configuration.yml`.
#### **PS**: Please note that all the columns' names should be lowercase.
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
  filter_rows : # This section is used to remove rows base on a given column.
    filter_column : # This argument defines the name of column to filter.
    value : # This argument defines the value used to remove rows.
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
  kfold: True # This argument selects whether to use Kfold split or not.
  kfold_column_name: fold # This argument specifies the column name which will hold the kfold splits if kfold is True.
  nfold: 5 # This argument defines the number of fold if kfold is True.
  train_val: True # This argument specifies whether to use train/validation split or not.
  train_val_name: validation  # This argument defines the column name which will hold the train/validation split if train_val is True.
  val_size: 0.1 # This argument defines the validation size.
```
