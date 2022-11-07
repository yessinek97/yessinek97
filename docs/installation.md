# Installation

Please clone the repository via SSH or HTTPs.

## Docker-based setup

### Docker Installation

1. Install Docker following the [official documentation](https://docs.docker.com/get-docker/).
2. For Linux, please execute the
   [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

### Mkdocs container

You can use Docker to open the Mkdocs server in the localhost machine to browser the documentation
pages locally.

```bash
# Build and run Mkdocs server
make docs-serve-dev
```

### IG Docker container

You can use Docker for training the models or inference on a separate dataset. There are three
stages for this pipeline to be functional.

First you need to build the Docker image:

```bash
# Build Docker Image
make ig_build
```

Then you need to create and start a container based on the image previously created:

```bash
# run the image
make ig_run
```

Finally, you need to enter the container shell in interactive mode

```bash
# Make script executable
make ig_bash
```

## Developers (`conda`-based setup)

Install `anaconda` if it is not already available on your system. Please follow the official
[installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

```
# Create a conda env
conda env create -f environment.ig.train.yaml && conda activate biondeep_ig_train
```

For local code development, **please make sure you install the pre-commit before committing any
changes.**

```
pre-commit install -t pre-commit -t commit-msg
```
