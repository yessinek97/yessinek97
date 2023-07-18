Applying features selection,training and evaluation on the same dataset can be launched with the `cimt` command where this command will execute a different commands in the following order:
- cimt-kfold-split (for features selection)
- cimt-features-selection
- cimt-kfold-split (for training)
- cimt-train

# cimt

- In Docker container:
```bash
cimt -d <train_data> -c <configuration_file_path> -n <experiment_name>
```
- In Conda environment:
```bash
python -m ig.main cimt -d <train_data> -c <configuration_file_path> -n <experiment_name>
```

Example:

- In Docker container:
```bash
cimt -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -c CIMT.yml -n BaseFeatures
```
- In Conda environment:
```bash
python -m ig.main cimt -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -c CIMT.yml -n BaseFeatures
```

```bash
Options:
  -c       TEXT    Data path to use [required]

  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the experiment [required]
```
# cimt-kfold-split

 Split the given data into a different splits train/test using kfold and the splits will be saved under the parent folder of the given data

- In Docker container:
```bash
cimt-kfold-split -d <train_data> -c <configuration_file_path>
```
- In Conda environment:
```bash
python -m ig.main cimt-kfold-split -d <train_data> -c <configuration_file_path>
```

Example:

- In Docker container:
```bash
cimt-kfold-split -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -c CIMT.yml
```
- In Conda environment:
```bash
python -m ig.main cimt-kfold-split -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -c CIMT.yml
```

```bash
Options:
  -d       TEXT     Data path to use [required]

  -c       TEXT     Configuration file to use [required]

  -t       Flag     If it's for train True and if it's features selection False   [Optional]
```
# cimt-features-selection

Launch the train command with features selection for each pair train/test dataset splitted by the `cimt-kfold-split` command and fetch the important features of each split in order to find the top N features overall.

- In Docker container:
```bash
cimt-features-selection -d <train_data_directory> -c <configuration_file_path> -n <experiment_name>
```
- In Conda environment:
```bash
python -m ig.main cimt-features-selection -d <train_data_directory> -c <configuration_file_path> -n <experiment_name>
```

Example:
- In Docker container:
```bash
cimt-features-selection -d ./data/CIMT2023Public/ -c CIMT.yml -n BaseFeatures
```
- In Conda environment:
```bash
python -m ig.main cimt-features-selection -d ./data/CIMT2023Public/ -c CIMT.yml -n BaseFeatures
```

```bash
Options:
  -c       TEXT    Folder path where the splitted data is located  [required]

  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the experiment [required]
```
# cimt-train

Train each pair train/test dataset splitted by the `cimt-kfold-split` command using the basic train command implemented in the framework and report the global topk and topk per split

- In Docker container:
```bash
cimt-train -d <train_data_directory> -c <configuration_file_path> -n <experiment_name>
```
- In Conda environment:
```bash
python -m ig.main cimt-train -d <train_data_directory> -c <configuration_file_path> -n <experiment_name>
```

Example:
```bash
cimt-train  -d ./data/CIMT2023Public/ -c  CIMT.yml -n BaseFeatures
```
```bash
Options:
  -d       TEXT    Folder path where the splitted data is located  [required]

  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the experiment [required]
```
# How to run cimt command
### 1- Download data
```bash
cd ig/
make bash
```
* Public dataset
```bash
pull --bucket_path gs://biondeep-data/IG/data/CIMT2023Public/PublicNetmhcpanGoRNA.csv --local_path data/CIMT2023Public/PublicNetmhcpanGoRNA.csv
```
* Features configuration
```bash
pull --bucket_path gs://biondeep-data/IG/data/CIMT2023Public/base_features.yml --local_path data/CIMT2023Public/base_features.yml
```
```bash
pull --bucket_path gs://biondeep-data/IG/data/CIMT2023Public/base_features_RNAGO.yml --local_path data/CIMT2023Public/base_features_RNAGO.yml
```
**PS**: You need to add the Google Storage  **Authentication credentials** either on **Conda** or on **Docker** path to be able to read data from GCP buckets paths.

```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/client_secret.json'
```
### 2- Configuration files
###  Main configuration file
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
- `CIMTFeaturesSelectionBaseFeatures.yml`/`CIMTFeaturesSelectionBaseFeaturesRNAGO.yml`: train configuration file for the features selection process
- `CIMTTrainingBaseFeatures.yml`/`CIMTTrainingBaseFeaturesRNAGO.yml`: train configuration file for the training  process

### 3- run cimt command

- In Docker container:
```bash
cimt -d <train_data> -c <configuration_file_path> -n <experiment_name>
```
- In Conda environment:
```bash
python -m ig.main cimt -d <train_data> -c <configuration_file_path> -n <experiment_name>
```

```bash
Options:
  -d       TEXT    data path  [required]

  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the experiment [required]
```
Example:

- In Docker container:
```bash
cimt -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -c CIMT.yml -n run_RNA
```
- In Conda environment:
```bash
python -m ig.main cimt -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -c CIMT.yml -n run_RNA
```

# Inference

- In Docker container:
```bash
cimt-inference -d <train_data> -n <experiment_name> -e -p
```
- In Conda environment:
```bash
python -m ig.main cimt-inference -d <train_data> -n <experiment_name> -e -p
```

```bash
Options:
  -d       TEXT    data path  [required]

  -n       TEXT    Name of the experiment [required]

  -e       Flag    Compute topk and print evaluation [Optional]

  -p       Flag    Process the input data [Optional]


```
Example:

- In Docker container:
```bash
cimt-inference -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -n run_RNA -e
```
- In Conda environment:
```bash
python -m ig.main cimt-inference -d ./data/CIMT2023Public/PublicNetmhcpanGoRNA.csv -n run_RNA -e
```

A dataset with the predictions will be saved under `model/<run_name>/<exp_name>/<data_name>.csv` and if the option **eval**  is on the scores of the evaluation will be saved in yml file under `model/<run_name>/file_name.yml`
