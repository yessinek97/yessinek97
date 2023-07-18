The goal is to find the best hyper-parameters for a specific model, for example Xgboost. You can use
the resulted params with the training command.

Additional parameters to the [global configuration file](./config.md#Global-configuration-file) are
found in the example file: `configuration/tune_xgboost.yml`

- In Docker container:
```bash
tune -train <train_data> -test <test_data> -c <configuration_file> -n <name>
```
- In Conda environment:
```bash
python -m ig.main tune -train <train_data> -test <test_data> -c <configuration_file> -n <name>
```

Example:

- In Docker container:
```bash
tune -train public.csv -test optima.tsv -c tune.yml -n test_tune
```
- In Conda environment:
```bash
python -m ig.main tune -train public.csv -test optima.tsv -c tune.yml -n test_tune
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Local or GCP path to the train/val data file [required]

  -test    PATH    Local or GCP path to the test data file [required]
```
