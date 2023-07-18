In order to evaluate the performance of a given score(e.g. binding, presentation score) you can
refer to the `compute_comparison_score` command.

- In Docker container:
```bash
compute-comparison-score -d <data_path>  -l <label_name>  -c  <column_name>
```

- In Conda environment:
```bash
python -m ig.main compute-comparison-score -d <data_path>  -l <label_name>  -c  <column_name>
```
