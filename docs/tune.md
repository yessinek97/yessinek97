The goal is to find the best hyper-parameters for a specific model, for example Xgboost. You can use
the resulted params with the training command.

Additional parameters to the [global configuration file](./config.md#Global-configuration-file) are
found in the example file: `configuration/tune_xgboost.yml`

```bash
tune -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>
```

Example:

```bash
tune -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c tune.yml -n  test_tune
```

```bash
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```
