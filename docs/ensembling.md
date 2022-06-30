The goal is to aggregate experiments using mean and output the topK results. Note: the experiments
should have had been tested on the same dataset.

There is two version:

- The first is by aggregating different experiments using the best checkpoint.
- The second is by aggregating the results of a one experiment from different feature selection
  methodds and using different models like (Xgboost/LGBM)

## Ensembling of different experiments

```bash
python -m biondeep_ig.main ensemblexprs -e /path/to/folder/models/
```

Example:

```bash
python -m biondeep_ig.main ensemblexprs -e sahin_cd8 -e sahin_tcga_cd8
```

```bash
Options:
  -e       TEXT  path to trained models [required]
```

You can also do the ensembling on the different models under the models directory like this:

```bash
python -m biondeep_ig.main ensemblexprs
```

Example:

```bash
python -m biondeep_ig.main ensemblexprs
```

## Ensembling one experiment

```bash
python -m biondeep_ig.main ensoneexp -s /path/to/a/model/
```

Example:

```bash
python -m biondeep_ig.main ensoneexp -s train_sahin_public
```

```bash
Options:
  -s       TEXT  path to trained model [required]
```

output: Topk results of the ensembling methods.
