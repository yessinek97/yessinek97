## Inference on a single model with a bunch of datasets:

You can launch Inference with a single test.

```
inference -d  <data_Test_paths>  -n <models_folder_name> -id <id_name>

inference -d test.csv  -n a_trained_model -id ID
```

```
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
  - id      TEXT    Unique id column [required]

```

## Eval trained models on a separate dataset

You can launch a compute metrics with a single test or multiple test sets.

```
compute-metrics -d  <data_Test_paths>  -n <models_folder_name>

compute-metrics -n a_trained_model -d   test1.tsv -d test2.tsv
```

```
Options:
  - d       TEXT    Path to dataset [required]
  - n       TEXT    Path to model folder name [required]
```
