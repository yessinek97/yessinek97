## Inference using on a single model

Inference using a single model on a given dataset is performed using the command.

```bash
inference -d  <data_Test_paths>  -n <models_folder_name> -id <id_name>
```

Example:

```bash
inference -d test.csv  -n a_trained_model -id ID
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
  - id      TEXT    Unique id column [required]
  - p       FLAG    Process the data or not


```

## Evaluation of trained models on given datasets

Evaluation metrics can be computed from a single or several test datasets using the command:

```bash
compute-metrics -d  <data_Test_paths>  -n <models_folder_name>
```

Exemple:

```bash
compute-metrics -n a_trained_model -d   test1.tsv -d test2.tsv
```

```bash
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
```
