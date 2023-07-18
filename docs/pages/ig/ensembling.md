The goal is to aggregate experiments using mean and output the topK results. Note: the experiments
should have had been tested on the same dataset.

There is two version:

- The first is by aggregating different experiments using the best checkpoint.
- The second is by aggregating the results of a one experiment from different feature selection
  methodds and using different models like (Xgboost/LGBM)

## Ensembling of different experiments

- In Docker container:
```bash
ensemblexprs -e /path/to/folder/models/
```
- In Conda environment:
```bash
python -m ig.main python -m ig.main ensemblexprs -e /path/to/folder/models/
```

Example:

- In Docker container:
```bash
ensemblexprs -e sahin_cd8 -e sahin_tcga_cd8
```
- In Conda environment:
```bash
python -m ig.main compute-comparison-score -d <data_path> -l <label_name>  -c  <column_name>
```


```bash
Options:
  -e       TEXT  path to trained models [required]
```

You can also do the ensembling on all different models under the `models` directory like this:

- In Docker container:
```bash
ensemblexprs
```
- In Conda environment:
```bash
python -m ig.main ensemblexprs
```

## Ensembling one experiment

- In Docker container:
```bash
ensoneexp -s /path/to/a/model/
```
- In Conda environment:
```bash
python -m ig.main ensoneexp -s /path/to/a/model/
```

Example:

- In Docker container:
```bash
ensoneexp -s train_sahin_public
```
- In Conda environment:
```bash
python -m ig.main ensoneexp -s train_sahin_public
```

```bash
Options:
  -s       TEXT  path to trained model [required]
```

output: Topk results of the ensembling methods.
