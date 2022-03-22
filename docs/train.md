## Train model

Training of a model can be launched with the `train` command using the provided configuration file,
the data file and a name. The parent directory from which the framework should be started is as
usual the /biondeep folder (not /biondeep/biondeep_ig).

```
train -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>

# Example
train -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c train.yml -n  test_train
```

```
Options:
  -c       TEXT    Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Path to the train/val data file [required]

  -test    PATH    Path to the test data file [required]
```

The trained models will be available in `biondeep_ig/models/<name>` directory
