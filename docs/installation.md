## Requirements

- Conda to setup the environment

## Conda environment creation

```
# Create a conda env
conda env create -f environment.ig.train.yaml && conda activate biondeep_ig_train

# Install pre-commit hooks (optional for developing)
pre-commit install -t pre-commit -t commit-msg
```
