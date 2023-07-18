# Installation

Please clone the repository via SSH or HTTPs.
```bash
git clone git@gitlab.com:instadeep/bioai-group/biondeep-structure.git
```

## Docker-based setup

### Docker Installation

1. Install Docker following the [official documentation](https://docs.docker.com/get-docker/).
2. For Linux, please execute the
   [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

### IG Docker container

You can use Docker for training the models or inference on a separate dataset. There are three
stages for this pipeline to be functional.

First you need to build the Docker image:

```bash
# Build Docker Image
make build
```

Then you need to create and start a container based on the image previously created:

```bash
# run the image
make run
```

Finally, you need to enter the container shell in interactive mode

```bash
# Make script executable
make bash
```

## Developers (`conda`-based setup)

Install `anaconda` if it is not already available on your system. Please follow the official
[installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

```
# Create a conda env
conda env create -f environment.ig.train.yaml && conda activate biondeep_ig
```

For local code development, if you don't already have Node.js, and npm installed, you'll need to install them properly.

```bash
# Installing pre-commit
sudo apt install pre-commit
```


```bash
# Installing nodejs and npm
sudo apt install -y nodejs npm
```

**please make sure you install the pre-commit before committing any
changes.**

```
pre-commit install -t pre-commit -t commit-msg
```

## Google Storage authentication

- You need to add the Google Storage [**Authentication credentials**](https://console.cloud.google.com/storage/browser/_details/biondeep-data/IG/biontech-tcr-16ca4aceba4c.json;tab=live_object?authuser=0) either on **Conda** or on **Docker** path to be able to read data from GCP buckets paths.

```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/client_secret.json'
```
