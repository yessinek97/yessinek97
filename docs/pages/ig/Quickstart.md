# Quickstart

First please make sure to follow [the installation steps](installation.md).

## 🚨 Important Notes

1. When running commands in **Conda environment (for developers)** you should first add `python -m ig.main` then your command `<IG_command>`. Example with `pull` data command:
    - In Docker container:

    ```bash
    pull --bucket_path gs://biondeep-data/IG/data/folder/file.csv --local_path /home/app/ig/data/folder/file.csv
    ```

    - In Conda environment:

    ```bash
    python -m ig.main pull --bucket_path gs://biondeep-data/IG/data/folder/file.csv --local_path /home/app/ig/data/folder/file.csv
    ```

2. Before running any command, please make sure the configuration file you are using contains the following parameters or directly use the predefined quickstart configuration files `configuration/quickstart_{operation}.yml`, with operation can be either **train**, **seed_fold** or **tune**:

    ```yml
    #process_data: False
    remove_unnecessary_folders: False
    trainable_features: features_quickstart.yml
    validation_strategy: True
    .
    .
    .
    force_features:
    - tested_score_biondeep_mhci
    - expression
    - tested_presentation_biondeep_mhci
    ```

## Downloading Data

### First option

In the first option, you can manually download the 3 data files one by one (train, test and features) from GCP using the available [`pull`](push_pull.md#docker-pull-command) command:

- Train data:

```bash
pull --bucket_path gs://biondeep-data/IG/data/quick_start/train.csv --local_path data/quick_start/train.csv
```

- Test data:

```bash
pull --bucket_path gs://biondeep-data/IG/data/quick_start/test.csv --local_path data/quick_start/test.csv
```

- Features configuration:

```bash
pull --bucket_path gs://biondeep-data/IG/data/quick_start/features_quickstart.yml --local_path data/quick_start/features_quickstart.yml
```

>:warning:  All these files `train.csv`, `test.csv`, `feature.yml` must be placed under the same folder.

### Second option

For this option, you can let the used command (`train`, `tune`, ...) take care of downloading the files automatically while running the command.

- For features configuration, make sure to specify the GCP link in `quickstart_train.yml`:

```yml
trainable_features: gs://biondeep-data/IG/data/quick_start/features_quickstart.yml
```

- For train and test files specify the GCP link in the command arguments:

```bash
train -train gs://biondeep-data/IG/data/quick_start/train.csv  -test gs://biondeep-data/IG/data/quick_start/test.csv  -n test_quick_start -c quickstart_train.yml
```

## Training

To launch the training script, you can run the command below:

```bash
train -train data/quick_start/train.csv -test data/quick_start/test.csv -n test_quick_start -c quickstart_train.yml
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

The trained models will be available in `models/<name>` directory.

## Tuning

If you want to use the tuning method:

```bash
tune -train data/quick_start/train.csv  -test data/quick_start/test.csv  -c quickstart_tune.yml -n  test_tune
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

## Seeds and Folds

You can run the model and train it with different seeds and folds.

```bash
train-seed-fold -train data/quick_start/train.csv  -test data/quick_start/test.csv  -c quickstart_seed_fold.yml -n  test_seed_fold
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

### Seeds and Folds Configuration

The configuration `configuration/quickstart_seed_fold.yml` contains the additional following keys:

```yaml
processing:
  seeds: [2058, 3058] # a list of random seed
  folds: [3, 4, 5] # a list of number of folds to be created. In case of single KFold experiment determines how many folds will be created.
```

## Inference

You can run the inference on a test data like this.

This inference example uses the model trained in the [`test_quick_start`](#training) experiment.

```bash
inference -d  data/quick_start/test.csv  -n test_quick_start -id ID
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
  - id      TEXT    Unique id column [required]
```

## Compute metrcis

Evaluation metrics can be computed from a single or several test datasets using the command:

```bash
compute-metrics -n test_quick_start -d   data/quick_start/test.csv
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]

```

## Ensembling

### Ensembling one experiment

```bash
ensoneexp -s test_quick_start/
```

```bash
Options:
  -s       TEXT  path to trained model [required]
```

output: Topk results of the ensembling methods.

>:warning: Ensembling multiple experiments is not working for the moment, the function will be changed entirely.

### Ensembling multiple experiments

You can run the ensemling of multiple experiments

```bash
ensemblexprs -e test_quick_start -e test_quick_start
```

```bash
Options:
  -e       TEXT  path to trained models [required]
```

You can also do the ensembling of all models under the same directory:

```bash
cd test_quick_start/
ensemblexprs
```
