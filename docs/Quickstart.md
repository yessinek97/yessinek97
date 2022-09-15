# Environment

Please clone the repository via SSH or HTTPs.

## Docker-based setup

### Docker Installation

1. Install Docker following the [official documentation](https://docs.docker.com/get-docker/).
2. For Linux, please execute the
   [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

### IG Docker container

You can use Docker for training the models or inference on a separate dataset. There are three
stages for this pipeline to be functional.

First you need to build the Docker image:

```bash
# Build Docker Image
make ig_build
```

Then you need to create and start a container based on the image previously created:

```bash
# run the image
make ig_run
```

Finally, you need to enter the container shell in interactive mode

```bash
# Make script executable
make ig_bash
```

# Important Note!

Before running any command, please make sure to change the config files accordingly:

- process_data: False
- remove_proc_data: false
- trainable_features: "features_Quickstart"
- validation_strategy: True

You need to foce the below features for training: force_features: - "tested_score_biondeep_mhci" -
"expression" - "tested_presentation_biondeep_mhci"

# Training

For training all you have to do it run the command below to test how the training is done:

```bash
 train -train Quickstart/train.csv  -test Quickstart/test.csv  -c  train_with_fs.yml -n  Quickstart_training
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

The trained models will be available in `models/<name>` directory.

# Tuning

If you want to use the tuning method:

```bash
 tune -train Quickstart/train.csv  -test Quickstart/test.csv  -c tune_configuration.yml -n  test_tune

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

```bash
 train-seed-fold -train Quickstart/train.csv  -test Quickstart/test.csv -c train_seed_fold.yml -n  test_seed_folds
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

## Configuration

The configuration `configuration/train_seed_fold.yml` contains the additional following keys:

```yaml
processing:
  seeds: [2058, 3058] # a list of random seed
  folds: [3, 4, 5] # a list of of number of folds to be created. In case of single KFold experiment determines how many folds will be created.
```

# Inference

You can run the inference on a test data like this: The inference uses the model trained in the
Quickstart_training experiment.

Example:

```bash
 inference -d Quickstart/test.csv  -n Quickstart_training -id ID
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
  - id      TEXT    Unique id column [required]

```

## Evaluation of trained models on given datasets

Evaluation metrics can be computed from a single or several test datasets using the command:

Exemple:

```bash
compute-metrics -n Quickstart_training -d  Quickstart/test.csv
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
```

# Ensembling

# Ensembling multiple experiments

You can run the ensemling of multiple experiments

```bash
ensemblexprs -e Quickstart_training -e Quickstart_training2
```

```bash
Options:
  -e       TEXT  path to trained models [required]
```

You can do the ensembling on the different models under the models directory like this:

```bash
ensemblexprs
```

# Ensembling one experiment

```bash
ensoneexp -s Quickstart_training/
```

```bash
Options:
  -s       TEXT  path to trained model [required]
```

output: Topk results of the ensembling methods.
