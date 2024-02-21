## Inference using on a single run

Inference using a single model on a given dataset is performed using the command.
PS: **Id** column must be included in the given data dataset.

- In Docker container:
```bash
inference -d <data_Test_paths> -n <models_folder_name> -id <id_name>
```
- In Conda environment:
```bash
python -m ig.main inference -d <data_Test_paths> -n <models_folder_name> -id <id_name>
```

Example:

- In Docker container:
```bash
inference -d test.csv -n a_trained_model -id ID
```
- In Conda environment:
```bash
python -m ig.main inference -d test.csv -n a_trained_model -id ID
```

```bash
Options:
  - d       TEXT    Loacal or GCP Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
  - id      TEXT    Unique id column [required]
```

## Evaluation of trained models on given datasets

Evaluation metrics can be computed from a single or several test datasets using the command:

- In Docker container:
```bash
compute-metrics -d <data_Test_paths> -n <models_folder_name>
```
- In Conda environment:
```bash
python -m ig.main compute-metrics -d <data_Test_paths> -n <models_folder_name>
```

Example:

- In Docker container:
```bash
compute-metrics -n a_trained_model -d test1.tsv -d test2.tsv
```
- In Conda environment:
```bash
python -m ig.main compute-metrics -n a_trained_model -d test1.tsv -d test2.tsv
```

```bash
Options:
  - d       TEXT    Local or GCP path to dataset [required]
  - n       TEXT    Path to model folder name [required]
```
