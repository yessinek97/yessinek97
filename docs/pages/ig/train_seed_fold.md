## Command

The goal is to train the model with a list of number of folds and a list of seeds.

- In Docker container:
```bash
train-seed-fold -train <train_data> -test <test_data> -c <configuration_file> -n <name>
```
- In Conda environment:
```bash
python -m ig.main train-seed-fold -train <train_data> -test <test_data> -c <configuration_file> -n <name>
```

Example:

- In Docker container:
```bash
train-seed-fold -train public.csv -test optima.tsv -c train_seed_fold.yml -n test_seed_folds
```
- In Conda environment:
```bash
python -m ig.main train-seed-fold -train public.csv -test optima.tsv -c train_seed_fold.yml -n test_seed_folds
```


```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Local or GCP path to the train/val data file [required]

  -test    PATH    Local or GCP path to the test data file [required]
```

## Configuration

The configuration `configuration/train_seed_fold.yml` contains the additional following keys:

```yaml
processing:
  seeds: [2058, 3058] # a list of random seed
  folds: [3, 4, 5] # a list of of number of folds to be created. In case of single KFold experiment determines how many folds will be created.
```
