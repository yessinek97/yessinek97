# CIMT

Applying features selection,training and evaluation on the same dataset can be launched with the `cimt` command where this command will execute a different commands in the following order:

1. **cimt-kfold-split**: split the given input data into N splits if `split_data` is True
2. **cimt-features-selection**: Train and evaluate on the splits and report Top 20 features using the weighting method
3. **cimt-kfold-split**: split the given input date into N splits for second training stage,if  `split_data` and `do_w_train` are True
4. **cimt-train**: Train and evaluate on the second splits using the Top20 weighted features if `do_w_train` is True

`split_data` and `do_w_train` are defined in the [main configuration](cimt.md#main-configuration-file) file.

## CIMT command
### Command usage
- In Docker container:
```bash
cimt -d <train_data> -dr <data_directory> -c <configuration_file_path> -n <experiment_name>
```
- In Conda environment:
```bash
python -m ig.main cimt -d <train_data>   -dr <data_directory> -c <configuration_file_path> -n <experiment_name>
```
```bash
Options:
  -d       TEXT    Data path to use [optional]

  -dr      TEXT    Folder path where the splitted data is located  [optional]

  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the experiment [required]
```
>🚨 Either data path **-d** or Folder path **-dr** should be given as argument
## How to run cimt command
The `cimt` command serves various purposes and can be utilized in the following manner:

  - The `cimt` command offers two types of input data:
      - **A CSV file**, the `cimt` command splits the data into N partitions (train/test) based on the configurations specified in the configuration file. To activate this feature, ensure that the `split_data` parameter is set to **True**. Failure to do so will result in an exception being raised.
      - **A directory** containing data splits (train/test) in the form of CSV files.
  - The cimt command provides two training methods:

    - **Feature selection and training** for each split: In this method, the command conducts feature selection and training individually for each split.

    - **Feature selection and training the top20 features** : This method involves feature selection and training for each split while employing a weighting method to identify the top 20 features across splits. Subsequently, the command trains different splits using these top 20 features. if `do_w_train` is set to **True** otherwise it will do **Feature selection and training** only.
### 1- Download data
- In Docker container:
```bash
pull --bucket_path gs://biondeep-data/IG/data/IG_24_03_2023/IG_24_03_2023Netmhcpan/cimt_2024/ --local_path ./data/cimt_2024
```
- In Conda environment:
```bash
python -m ig.main pull --bucket_path gs://biondeep-data/IG/data/IG_24_03_2023/IG_24_03_2023Netmhcpan/cimt_2024/ --local_path ./data/cimt_2024
```
This command will download all available data for CIMT:

    - **Public**: folder contains the clean version of the public dataset.
    - **CIMT_2024_GeneBased**: folder contains different train/test splits for the public data based on gene.
    - **CIMT_2024_Random**: folder contains different train/test splits for the public data with random splits.

    **PS**: You need to add the Google Storage  **Authentication credentials** either on **Conda** or on **Docker** path to be able to read data from GCP buckets paths. [GCP Authentication steps](installation.md#google-storage-authentication)
### 2- Configuration files
```yaml
General:
  split_data: # split data configuration
  do_w_train: # train using weighted features or not
  eval:
    metrics: # evaluation metrics
      - topk
      - logloss
      - roc
    comparison_columns: # list of the columns which will be compared with  model outcome
    label_column: # label column

  features_selection: # features selection configuration
    test_file_pattern: # test file pattern with place holder for split number if dr argument is provided (e.g., test_{}.csv)
    train_file_pattern: # train  file pattern with place holder for split number if dr argument is provided (e.g., train_{}.csv)
    n_splits:  # number os splits
    seed:  # seed which will be used to split the train data
    Ntop_features: # number of the final selected  features
    default_configuration: # default configuration for features selection process
  w_train: # configuration for the train using weighted features
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
    w_train :
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
  Predefined configuration files are implemented under the configuration file:

    - `CIMT.yml`: main configuration file
    - `CIMTFeaturesSelection.yml`: train configuration file for the features selection process
    - `CIMTTraining.yml`: train configuration file for the training with weighted features process
### 3- Run Cimt command
#### From file
- In Docker container:
```bash
cimt -d ./data/cimt_2024/public/public_cimt.csv -c CIMT.yml -n run
```
- In Conda environment:
```bash
python -m ig.main cimt -d ./data/cimt_2024/public/public_cimt.csv -c CIMT.yml -n run
```
#### From data directory contains different data splits
- In Docker container:
```bash
cimt -dr ./data/cimt_2024/cimt_2024_random -c CIMT.yml -n run
```
- In Conda environment:
```bash
python -m ig.main cimt -dr ./data/cimt_2024/cimt_2024_random  -c CIMT.yml -n run
```
