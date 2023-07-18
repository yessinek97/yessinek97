## Inference using on a multiple  runs

multi-inference  is used to compute the predication of the different runs or the output of the `multi-train` command
PS: **Id** column must be included in the given data dataset.

- In Docker container:
```bash
multi-inference -d <data_Test_paths> -mn <multi_train_run_name>  -sn <single_train_names> -id <id_name> -l label_name -p -e
```
- In Conda environment:
```bash
python  ig.main multi-inference -d <data_Test_paths> -mn <multi_train_run_name>  -sn <single_train_names> -id <id_name> -l label_name -p -e
```
```bash
Options:
  - d       TEXT[Multiple]    Loacal or GCP Path to dataset [required]
  - mn      TEXT              Path to the multiple train folder [required]
  - sn      TEXT[Multiple]    Name of the single run folder under the multi-train folder
                              which will be used for the inference
  - id      TEXT              Unique id column [required] default "id"
  - p       FLAG              Process the data or not [Optional]
  - l       TEXT              Label name [Optional]
  - e       FLAG              Eval data or not if True the label name should be defined [Optional]


```
Example:

- In Docker container:
```bash
multi-inference -d test_1.csv -d test_2.csv -mn multi_train_run_name -sn run1 -sn run2 -id ID
OR
multi-inference -d test_1.csv -d test_2.csv -mn multi_train_run_name -id ID
OR
multi-inference -d test_1.csv -d test_2.csv -mn multi_train_run_name -id ID -e -l cd8_any
```

- In Conda environment:
```bash
python -m ig.main multi-inference -d test_1.csv -d test_2.csv -mn multi_train_run_name -id ID
```
