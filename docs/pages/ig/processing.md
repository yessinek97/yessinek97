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
 - Process train and other main datasets: takes the train data(Public table) and other main datasets  (Optima,Sahin,..) to do the processing and return the common features between the train and the other datasets.
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

This example illustrates how the configuration file should be. You can refer to **Processing configuration file section** at the [Data configuration file documentation page](data_configuration.md#preprocessing) for more details.

### Example
```yml
data_version: IG_16_11_2022
data_type: Netmhcpan
push_gcp: False
processing:
  legend:
    # path : #legend.csv
    filter_column: "to_take"
    value: 1
    feat_col_name: features
  # filter_rows :
  #   filter_column :
  #   value :
  label: cd8_any
  id: id
  ids:
    - id
    - patientid
    - wt_27mer
    - mut_27mer
    - tested_peptide
    - wildtype_peptide
    - author/source
    - genename
  proxy_model_columns:
    proxy_m_peptide: tested_peptide_netmhcpan4.1
    proxy_wt_peptide: wildtype_peptide_netmhcpan4.1
    proxy_allele: allele_netmhcpan4.1
    scores:
      tested_score_netmhcpan4.1: proxy_score_presentation
      tested_best_rank_netmhcpan4.1: proxy_score_rank_presentation
      tested_ba_score_netmhcpan4.1: proxy_score_score_binding
      tested_ba_rank_netmhcpan4.1: proxy_score_rank_binding
  expression:
    filter:
      - exp
      - tcg
      - tpm
      - gtex
    raw_name: expression_for_model
    name: expression
  exclude_features:
    - cd4
    - cd8
    - any
    - wt
    - wildtype
    - nontest
    - prime_rank
    - prime_score
    - length_tested_peptide
  include_features:
    - proxy_score
    - rnalocalization
    - go_term_rna
    - go_term_.*_embed
    - pathway
    - deeptap
    - specificity_gtex

  keep_include_features: False
  nan_ratio: 0.6

features:
  file_name: features.yml
  float: True
  include_features_float: True
  int: True
  include_features_int:
  categorical: False
  include_features_categorical: True
  bool: False
  include_features_bool: True
split:
  kfold: True
  kfold_column_name: fold
  nfold: 5
  train_val: True
  train_val_name: validation
  val_size: 0.1
  source_split: True
  source_column: author/source

```
## Examples

In order to execute the processing command we need to define the processing configuration file(as shown in the configuration section) or there is a pre-defined configuration file under the configuration directory `processing_configuration.yml` ready to use,and Thanks to the implemented [Pull command]  it's possible to download data from GCP

### Train data only
* [Pull](push_pull.md#push-pull-command) Public from gcp
```bash
pull --bucket_path gs://biondeep-data/IG/BntPipelineData/IG_24_03_2023/Processing/raw_data/NetMHCpan/publicMUT_20230324_v4_NetMHCpan.tsv --local_path data/row_data
```
* Execute the processing command
```bash
processing -t ./data/row_data/publicMUT_20230324_v4_NetMHCpan.tsv  -c processing_configuration.yml
```

### Process train and other main datasets
* [Pull](push_pull.md#push-pull-command) Public and Optima from gcp
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
* [Pull](push_pull.md#push-pull-command) Other data(OptimaPD) from gcp
```bash
pull --bucket_path gs://biondeep-data/IG/BntPipelineData/IG_24_03_2023/Processing/raw_data/NetMHCpan/optimaPD_20210825_NetMHCpan.tsv --local_path data/row_data
```
* Execute the processing command
```bash
processing -odp ./data/row_data/optimaPD_20210825_NetMHCpan.tsv -odn optimaPD -o proc_data_IG_16_11_2022_Netmhcpan
```
