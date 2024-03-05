# Installation

Please clone the [`biondeep-IG`](https://gitlab.com/instadeep/biondeep-ig) repository via SSH or HTTPs.

```bash
git clone git@gitlab.com:instadeep/biondeep-ig.git
```

Then you can choose to work using the [Docker based setup](#docker-based-setup) or the [Conda based setup (for developers)](#conda-based-setup-for-developers)

## Docker based setup

### Docker Installation

1. Install `Docker` following the [official documentation](https://docs.docker.com/engine/install/ubuntu/).

2. For Linux, please execute the
   [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

3. Install [the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) and [configure docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) (**don’t use** rootless mode!)

4. Then [verify that your installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html#running-a-sample-workload-with-docker) is working correctly

5. (Optional for developers) If you are using VS Code you should install the `Docker` extension and `Remote Development` Extension Pack which will allow you to [attach VS Code to a running container](https://code.visualstudio.com/docs/devcontainers/attach-container) and work inside the container.

### Install biondeep-IG Docker container

You can use `Docker` for training the models or inference on a separate dataset. There are three
stages for this pipeline to be functional.

* First, you need to open a new terminal inside the `biondeep-ig` folder

    ```bash
    cd path/to/biondeep-ig
    ```

* Next, you need to __build__ the Docker image then __create__ and __start__ the IG container.

    ```bash
    make run
    ```

    >* Building the docker image will ask you to login to [GitLab container registry](https://gitlab.com/instadeep/biondeep-ig/container_registry/) with the rights to `read_registry` and `write_registry`.
    >* You can create a [new access token here](https://gitlab.com/instadeep/biondeep-ig/-/settings/access_tokens) and use it as a login password.

* Finally, you need to __open the container shell__ in interactive mode. This step will also make sure the repo is installed in the container environment.

    ```bash
    make bash
    ```

* (Optional) You can also [attach VS Code to the running container](https://code.visualstudio.com/docs/devcontainers/attach-container) and work inside the container.

## Conda based setup (for developers)

1. __Install `anaconda`__ if it is not already available on your system. Please follow the official
[installation instructions](https://docs.anaconda.com/free/anaconda/install/).

2. Open a new terminal inside the `biondeep-ig` folder

      ```bash
      cd path/to/biondeep-ig
      ```

3. __Create__ and __activate__ the conda environment

      ```bash
      conda env create -f .environment-dev.yaml && conda activate biondeep_ig
      ```

4. If you are a contributor, please make sure to install `pre-commit` before committing any changes.

      ```bash
      pre-commit install -t pre-commit -t commit-msg
      ```

5. (Optional) If you have trouble with `pre-commit install` that is related to `Node.js` and/or `npm`, then try to install them using:

      ```bash
      sudo apt install -y nodejs npm
      ```

## Google Storage authentication

In order to be able to Read & Write [DATA](https://console.cloud.google.com/storage/browser/biondeep-data/IG/data) and [Models](https://console.cloud.google.com/storage/browser/biondeep-data/IG/experiments) from the BioNdeep-IG GCP storage Buckets:

1. You have to ask a developer to send you the `Authentication credentials file`. This file must be added at the root of the project's folder.

2. Export the path to the GCP `Authentication credentials file` in your local environment or inside the docker container (must be repeated in each new terminal)

      ```bash
      export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/json_file.json'
      ```

>Read the [Push & Pull command](push_pull.md) documentation for more details about reading and writing from the GCP buckets.
