## Inference on a single model with a bunch of datasets TODO:

## Eval trained models on a seperate dataset

You can launch a trainon with a single test or multiple test sets.

```
python -m  biondeep_ig.main compute-metrics -test  <data_Test_paths>  -n <models_folder_name>

python -m biondeep_ig.main compute-metrics -n a_trained_model -test   test1.tsv -test test2.tsv
```

```
Options:
  -Test       TEXT    Path to dataset [required]
  - n         TEXT    path tomodel folder name [required]
```
