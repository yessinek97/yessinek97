## Ablation study

### Training models

The performance of the IG models can be compared with respect to the selection of certain
pre-defined features. Typically the main features are:

- 18 BNT features
- binding score,
- expression score,
- presentation score.

An IG model can be trained separately for various combinations of these features with the following
command:

```bash
ablation_study -train <train_data_path> -test <test_data_path> -n <ablation_study_folder_name> -c <configuration_file>
```

Parallel execution is possible by adding `--nprocs <number_of_threads>`.

### Post-processing

Once all combinations have been run, a summary of the best models performance can be displayed with
the following command:

```bash
ablation_post -n <ablation_study_folder_name>
```
