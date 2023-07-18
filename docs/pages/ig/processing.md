# Data Processing
In order to use the available training command in the IG framework, additional processing of the BNT data pipeline output (raw data) is necessary to ensure good data quality by :
- Removing the missing label
- Removing duplicated features
- Removing features with more than a specific threshold of missing values
- Remove features with only one unique value
- Process the `expression` columns
 Finally, only the common features between the train and the main datasets are processed and retained.

## Processing command

 The processing command is used in three different ways :

 - Process train only: takes only the train data (public table) to do the processing and  return  the available features.
 - Process train and other main datasets: takes the train data(Public table) and other main datasets  (Optima,Sahin,..) to do the processing and return the commune features between the train and the other datasets.
 - Apply previous processing  to another dataset: takes another datasets (Optimapd,Sahin,...) and apply processing based on previous  processing for train and other main data.
```bash
processing -t <train_path>
           -mdp <first_main_data_path> -mdn <first_main_data_name>
           -odp <first_new_data_path> -odn <first_other_data_name>
           -c <configuration_file_name>
           - o <output_directory_name>
           -ignore
```
```bash
Options:
    -t        TEXT              train data path
    -mdp      TEXT [Multiple]   main data path
    -odp      TEXT [Multiple]   other data path
    -mdn      TEXT [Multiple]   main data name
    -odn      TEXT [Multiple]   other data name
    -c        TEXT              processing configuration file name
    -o        TEXT              output directory where the processed data and metadata are saved
    -ignore   Flag              ignore missing features while processing other data
```
## Execution

 As Mentioned above there are three ways of execution of the processing command
### Train data only
 ```bash
processing -t <train_path> -c <configuration_file_name>
```
```bash
Options:
    -t        TEXT       train data path
    -c        TEXT       processing configuration file name

```
### Process train and other main datasets
```bash
processing -t <train_path>
           -mdp <first_main_data_path> -mdn <first_main_data_name>
           -c <configuration_file_name>
```
```bash
Options:
    -t        TEXT              train data path
    -mdp      TEXT [Multiple]   main data path
    -mdn      TEXT [Multiple]   main data name
    -c        TEXT              processing configuration file name
```
### Apply  processing another data
```bash
processing -odp <first_other_data_path> -odn <first_other_data_name>
           - o <output_directory_name>
           -ignore
```
```bash
Options:

    -odp      TEXT [Multiple]   other data path
    -odn      TEXT [Multiple]   other data name
    -o        TEXT              output directory where the processed
                                data and metadata are saved
    -ignore   Flag              ignore missing features while
                                processing other data
```
NB: the output of the processing command are saved under the directory `data/data_proc_{version}_{type}` where `version`  and `type` are defined in the configuration file
## Configuration
```yml
data_version: # Version of the data
data_type: # Data type :Netmhcpan or BionNDeep, use it  with data_version to create the output directory name "data_proc_{data_version}_{data_type}
push_gcp:  # whether push the outcome to gcp bucket or not (True or False)
processing: # processing section
  legend: # legend section use to filter features
    path : #legend file name,where the file should be  under the parent directory of the train data
    filter_column: # column'name used to filter features with
    value: # the value inside the filter_column which represent the feature to take
    feat_col_name: #colum's name which hold the features name
  filter_rows : # use to remove rows base on a given column
    filter_column : # name of column
    value : # value use to remove rows
  label: cd8_any #label name
  id: id # the unique id
  ids: # list of the columns will be saved with the proceeded data
    - id
    - patientid
    - wt_27mer
    - mut_27mer
    - tested_peptide
    - wildtype_peptide
    - author/source
    - genename
  proxy_model_columns: # Binder/presentation model section
    proxy_m_peptide: tested_peptide_netmhcpan4.1 # name of the best binder mutated peptide column
    proxy_wt_peptide: wildtype_peptide_netmhcpan4.1 # the associated WT peptide for the best binder mutated peptide column
    proxy_allele: allele_netmhcpan4.1 # best allele column name
    scores: # list of the available  Binder/presentation  scores and the associated new name
      # real_score_name : new_name with a prefix e.g. proxy_score
      tested_score_netmhcpan4.1: proxy_score_presentation
      tested_best_rank_netmhcpan4.1: proxy_score_rank_presentation
      tested_ba_score_netmhcpan4.1: proxy_score_score_binding
      tested_ba_rank_netmhcpan4.1: proxy_score_rank_binding
  expression: # expression section
    filter: #list of the keyword to filter all the expression columns
      - exp
      - tcg
      - tpm
      - gtex
    raw_name: expression_for_model #  expression's name to use
    name: expression # new name of the raw_name
  exclude_features: # list of excluded features by keyword or regex format
    - cd4
    - cd8
    - any
    - wt
    - wildtype
    - nontest
    - prime_rank
    - prime_score
    - length_tested_peptide
  include_features: # list of included features by keyword or regex format
    - proxy_score
    - rnalocalization
    - go_term_rna
    - go_term_.*_embed
    - pathway
    - deeptap
    - specificity_gtex
  keep_include_features: False # whether force processing to keep features from include list in case they will be  removed during processing
  nan_ratio: 0.6 #ratio on nan values
features: # features section
  file_name: features.yml # name of features configuration file
  float: True # whether choose to use float features type or not
  include_features_float: True # whether force processing  to keep the intercession between float type and include features in case of float = False
  int: True  # whether choose to use int features type or not
  include_features_int: True # whether force processing  to keep the intercession between int type and include features in case of int = False
  categorical: False # whether choose to use categorical features type or not
  include_features_categorical: True # whether force processing  to keep the intercession between categorical type and include features in case of categorical = False
  bool: False # whether choose to use bool features type or not
  include_features_bool: True # whether force processing  to keep the intercession between bool type and include features in case of bool = False
split: #Corss validation section
  kfold: True # whether use Kfold split or not
  kfold_column_name: fold # the column name which will hold the kfold splits if kfold is True
  nfold: 5 # number of fold if kfold is True
  train_val: True # whether use train/validation split or not
  train_val_name: validation  # the column name which will hold the train/validation split if train_val is True
  val_size: 0.1 # validation size
```
## Examples

In order to execute the processing command we need to define the processing configuration file(as shown in the configuration section) or there is a pre-defined configuration file under the configuration directory `processing_configuration.yml` ready to use,and Thanks to the implemented [Pull command]  it's possible to download data from GCP

### Train data only
* [Pull]((https://instadeep.gitlab.io/bioai-group/biondeep-structure/ig/push_pull/#the-pull-command)) Public from gcp
```bash
pull --bucket_path gs://biondeep-data/IG/BntPipelineData/IG_24_03_2023/Processing/raw_data/NetMHCpan/publicMUT_20230324_v4_NetMHCpan.tsv --local_path data/row_data
```
* Execute the processing command
```bash
processing -t ./data/row_data/publicMUT_20230324_v4_NetMHCpan.tsv  -c processing_configuration.yml
```

### Process train and other main datasets
* [Pull]((https://instadeep.gitlab.io/bioai-group/biondeep-structure/ig/push_pull/#the-pull-command)) Public and Optima from gcp
```bash
pull --bucket_path gs://biondeep-data/IG/BntPipelineData/IG_24_03_2023/Processing/raw_data/NetMHCpan/publicMUT_20230324_v4_NetMHCpan.tsv --local_path data/row_data
```
```bash
pull --bucket_path gs://biondeep-data/IG/BntPipelineData/IG_24_03_2023/Processing/raw_data/NetMHCpan/optima_20210831_NetMHCpan.tsv --local_path data/row_data
```
* Execute the processing command
```bash
processing -t ./data/row_data/publicMUT_20230324_v4_NetMHCpan.tsv  -mdp ./data/row_data/optima_20210831_NetMHCpan.tsv -mdn optima  -c processing_configuration.yml
```
### Apply  processing another data
* [Pull]((https://instadeep.gitlab.io/bioai-group/biondeep-structure/ig/push_pull/#the-pull-command)) Other data(OptimaPD) from gcp
```bash
pull --bucket_path gs://biondeep-data/IG/BntPipelineData/IG_24_03_2023/Processing/raw_data/NetMHCpan/optimaPD_20210825_NetMHCpan.tsv --local_path data/row_data
```
* Execute the processing command
```bash
processing -odp ./data/row_data/optimaPD_20210825_NetMHCpan.tsv -odn optimaPD -o proc_data_IG_16_11_2022_Netmhcpan
```
