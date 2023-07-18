Training of a single experiment can be launched with the `train` command using the provided configuration file,
the data file and a name. The parent directory from which the framework should be started is as
usual the /ig folder.

- In Docker container:
```bash
train -train <train_data> -test <test_data> -c <configuration_file> -n <name>
```
- In Conda environment:
```bash
python -m ig.main train -train <train_data> -test <test_data> -c <configuration_file> -n <name>
```

Example:

- In Docker container:
```bash
train -train public.csv  -test optima.tsv -c train.yml -n  test_train
```
- In Conda environment:
```bash
python -m ig.main train -train public.csv  -test optima.tsv -c train.yml -n  test_train
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Local or GCP path to the train/val data file locally or from GCP  [required]

  -test    PATH    Path to the test data file locally or from GCP [required]
```

The trained models will be available in `models/<name>` directory.
