## Modular training

The goal is to train multiple models each with a separate set of features, i.e. feature contraction.
Finally, a linear model will be trained on the predictions of each separate model to predict
immunogenicity. An example of a config file for this model type is given in
configuration/final_configuration_modular_train.yml

```bash
modulartrain -train <train_data>  -test <test_data>  -c  <configuration_file> -n  <name>
```

Example:

```bash
python -m  biondeep_ig.main modulartrain -train publicIEDBFilteredTransformerS128_20210827_out.csv  -test optimaAnonymizedimmunogenicityTransformerS128_20210818_out.tsv  -c modular_train.yml -n  test_modulartrain
```
