# Environment

Please clone the repository via SSH or HTTPs.

## Setup

### Conda-based setup

Please refer to [Developers (conda-based setup)](https://instadeep.gitlab.io/bioai-group/biondeep-structure/ig/installation/#developers-conda-based-setup).

### Docker-based setup

#### Docker Installation

1. Install Docker following the [official documentation](https://docs.docker.com/get-docker/).
2. For Linux, please execute the
   [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

#### IG Docker container

You can use Docker for training the models or inference on a separate dataset. There are three
stages for this pipeline to be functional.

First you need to build the Docker image:

```bash
cd ig
# Build Docker Image
make build
```

Then you need to create and start a container based on the image previously created:

```bash
# run the image
make run
```

Finally, you need to enter the container shell in interactive mode

```bash
# Make script executable
make bash
```

# Important Note!

Before running any command, please make sure to change the config files accordingly:
```yml
- process_data: False
- remove_unnecessary_folders: False
- trainable_features: features_quickstart.yml
- validation_strategy: True
```

You need to force the below features for training:

```yml
force_features:
- tested_score_biondeep_mhci
- expression
- tested_presentation_biondeep_mhci
```
or use directly the defined configuration file `configuration/quickstart_train.yml`

export the GCP credentals inside the ig-container
```bash
cd ig
export GOOGLE_APPLICATION_CREDENTIALS=CREDENTIALS_FILE.json
```
# Training
## First option
In the first option you need to download 3 files using the available [`pull`](https://instadeep.gitlab.io/bioai-group/biondeep-structure/ig/push_pull/#the-pull-command) command:

Train data:
- In Docker container:
```bash
pull --bucket_path gs://biondeep-data/IG/data/quick_start/train.csv --local_path data/quick_start/train.csv
```
- In Conda environment:
```bash
python -m ig.main pull --bucket_path gs://biondeep-data/IG/data/quick_start/train.csv --local_path data/quick_start/train.csv
```
Test data:
- In Docker container:
```bash
pull --bucket_path gs://biondeep-data/IG/data/quick_start/test.csv --local_path data/quick_start/test.csv
```
- In Conda environment:
```bash
python -m ig.main pull --bucket_path gs://biondeep-data/IG/data/quick_start/test.csv --local_path data/quick_start/test.csv
```
Features configuration:
- In Docker container:
```bash
pull --bucket_path gs://biondeep-data/IG/data/quick_start/features_quickstart.yml --local_path data/quick_start/features_quickstart.yml
```
- In Conda environment:
```bash
python -m ig.main pull --bucket_path gs://biondeep-data/IG/data/quick_start/features_quickstart.yml --local_path data/quick_start/features_quickstart.yml
```
**All this file should be placed under one directory**

For training all you have to do it run the command below to test how the training is done:

- In Docker container:
```bash
train -train data/quick_start/train.csv -test data/quick_start/test.csv -n test_quick_start -c quickstart_train.yml
```
- In Conda environment:
```bash
python -m ig.main train -train data/quick_start/train.csv -test data/quick_start/test.csv -n test_quick_start -c quickstart_train.yml
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

The trained models will be available in `models/<name>` directory.
## Second option
For this option no need to download any file the `train` command will take care of it
Before running any command, please make sure to change the config file accordingly:

```yml
- trainable_features: gs://biondeep-data/IG/data/quick_start/features_quickstart.yml
```
For training all you have to do it run the command below

- In Docker container:
```bash
train -train gs://biondeep-data/IG/data/quick_start/train.csv  -test gs://biondeep-data/IG/data/quick_start/test.csv  -n test_quick_start -c quickstart_train.yml
```
- In Conda environment:
```bash
python -m ig.main train -train gs://biondeep-data/IG/data/quick_start/train.csv  -test gs://biondeep-data/IG/data/quick_start/test.csv  -n test_quick_start -c quickstart_train.yml
```

# Tuning

If you want to use the tuning method:
- In Docker container:
```bash
tune -train data/quick_start/train.csv  -test data/quick_start/test.csv  -c quickstart_tune.yml -n  test_tune
```
- In Conda environment:
```bash
python -m ig.main tune -train data/quick_start/train.csv  -test data/quick_start/test.csv  -c quickstart_tune.yml -n  test_tune
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

# Seed and Folds

You can run the model and train it with different seeds and folds.
- In Docker container:
```bash
train-seed-fold -train data/quick_start/train.csv  -test data/quick_start/test.csv  -c quickstart_seed_fold.yml -n  test_seed_fold
```
- In Conda environment:
```bash
python -m ig.main train-seed-fold -train data/quick_start/train.csv  -test data/quick_start/test.csv  -c quickstart_seed_fold.yml -n  test_seed_fold
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

## Configuration

The configuration `configuration/train_seed_fold_quickstart.yml` contains the additional following keys:

```yaml
processing:
  seeds: [2058, 3058] # a list of random seed
  folds: [3, 4, 5] # a list of of number of folds to be created. In case of single KFold experiment determines how many folds will be created.
```

# Inference

You can run the inference on a test data like this: The inference uses the model trained in the
`test_quick_start` experiment.

Example:


- In Docker container:
```bash
inference -d  data/quick_start/test.csv  -n test_quick_start -id ID
```
- In Conda environment:
```bash
python -m ig.main inference -d  data/quick_start/test.csv  -n test_quick_start -id ID
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
  - id      TEXT    Unique id column [required]
  - p       FLAG    Process the data or not [optional]

```

# Compute metrcis

Evaluation metrics can be computed from a single or several test datasets using the command:

Example:

- In Docker container:
```bash
compute-metrics -n test_quick_start -d   data/quick_start/test.csv
```
- In Conda environment:
```bash
python -m ig.main compute-metrics -n test_quick_start -d   data/quick_start/test.csv
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
  - p       FLAG    Process the data or not [optional]

```

# Ensembling

## Ensembling multiple experiments

You can run the ensemling of multiple experiments

- In Docker container:
```bash
ensemblexprs -e test_quick_start -e test_quick_start
```
- In Conda environment:
```bash
python -m ig.main ensemblexprs -e test_quick_start -e test_quick_start
```

```bash
Options:
  -e       TEXT  path to trained models [required]
```

You can do the ensembling on the different models under the models directory like this:

```bash
ensemblexprs
```

## Ensembling one experiment

- In Docker container:
```bash
ensoneexp -s test_quick_start/
```
- In Conda environment:
```bash
python -m ig.main ensoneexp -s test_quick_start/
```

```bash
Options:
  -s       TEXT  path to trained model [required]
```

output: Topk results of the ensembling methods.
