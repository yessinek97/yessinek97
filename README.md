# BionDeep IG

## Installation

**Requirements:**

- docker 19.03 or later with [buildx](https://docs.docker.com/buildx/working-with-buildx/#install)
- [make](https://www.gnu.org/software/make/)
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or
  [miniconda](https://docs.conda.io/en/latest/miniconda.html)

**Step-by-step:**

1. Install the local conda environment with pre-commit:

```
conda env create -f .environment_dev.yaml
conda activate biondeep_ig
(biondeep_ig) pre-commit install -t pre-commit -t commit-msg
```

2. Build and run your container

```
make bash
```
