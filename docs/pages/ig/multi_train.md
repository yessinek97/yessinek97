# multi-train
Training of multiple experiments can be launched with the `multi-train` command using the provided configuration files,
the data file and a name. The parent directory from which the framework should be started is as
usual the /ig folder.

- In Docker container:
```bash
multi-train -train <train_data> -test <test_data> -c <multi_train_configuration_file> -n <name> -dc <default_configuration_file>
```
- In Conda environment:
```bash
python -m ig.main multi-train -train <train_data> -test <test_data> -c <multi_train_configuration_file> -n <name> -dc <default_configuration_file>
```

Example:

- In Docker container:
```bash
multi-train -train public.csv -test optima.tsv -c multi_train_configuration.yml -dc default_configuration.yml -n multi_train_exp
```
- In Conda environment:
```bash
python -m ig.main multi-train -train public.csv -test optima.tsv -c multi_train_configuration.yml -dc default_configuration.yml -n multi_train_exp
```

```bash
Options:
  -c       TEXT    multiple train Configuration file to use [required]

  -dc      TEXT    default Configuration file to use [required]

  -n       TEXT    Name of the model run [required]

  -train   PATH    Local or GCP path to the train/val data file locally or from GCP  [required]

  -test    PATH    Local or GCP path to the test data file locally or from GCP [required]
```

The trained models will be available in `models/<name>` directory.

## Configuration file

In order to run multiple experiments command an additional configuration is required.
A default multiple training configuration file is provided at `configuration/multi_train_configuration.yml`.

In the following more details on the parameters are provided.
```yaml

  experiments:
    experiment_name_1: # the name of the experiment
      section_to_change : #the section name which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
        parameter_1 :  #section's parameter which will be modified
        parameter_2 :
      section_to_change : #the section name  which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
        parameter_1 :  #section's parameter which will be modified
        parameter_2 :

    experiment_name_2: # the name of the experiment
      section_to_change : #the section name which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
        parameter_1 :  #section's parameter which will be modified
        parameter_2 :
      section_to_change : #the section name  which will be changed on the default configuration file(e.g.,experiments, label, processing, evaluation,... )
        parameter_1 :  #section's parameter which will be modified
        parameter_2 :

  params:
   metrics: # list of metrics which will be displayed in the end

```
Example:
```yaml
experiments:
  BNT: # first experiment name
    processing:
      trainable_features: gs://biondeep-data/IG/data/IG_16_11_2022Biondeep/ features_IG_16_11_2022_ensemble_expression_presentation_bnt.yml
    FS:
      force_features :
        - expression

  BNT_Biondeep: # second experiment name
    processing:
      trainable_features: gs://biondeep-data/IG/data/IG_16_11_2022Biondeep/features_IG_16_11_2022_ensemble_expression_presentation_bnt_biondeep.yml
    FS:
      force_features :
        - expression
        - tested_score_biondeep_mhci
        - tested_pres_score_biondeep_mhci
params:
  metrics:
    - topk
    - topk_20_patientid
```
