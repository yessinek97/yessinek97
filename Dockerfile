# To replace with the following image once it is available on the registry
# FROM registry.gitlab.com/instadeep/biondeep-ig/py-pyrosetta-rosetta-tmlgn:38-443f-313
FROM registry.gitlab.com/instadeep/bio-gnn/py-pyrosetta-rosetta-tmlgn:38-443f-313 AS ci

# Set different env variables linked to TF
# Do not take all the GPUs memory by default
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ID of GPUs listed by TF matches the output of nvidia-smi
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
# Filter the warnings from TF
ENV TF_CPP_MIN_LOG_LEVEL=3

# By default ${HOME_DIRECTORY}/.local is used
# This directory is erased if we map the ${HOME_DIRECTORY} to local ${PWD}
ENV PYTHONUSERBASE=/.pip_packages

# Update and upgrade your base image
RUN apt-get update && \
        apt-get upgrade -y

# Install required system dependencies and clear cache
RUN DEBIAN_FRONTEND=noninteractive apt-get install git -y && \
        apt-get clean

# Copy the requirements file into /tmp directory
COPY ./requirements.txt /tmp/requirements.txt

# Install python requirements
RUN pip install --upgrade --quiet pip setuptools && \
        pip install --no-cache-dir -r /tmp/requirements.txt && \
        rm /tmp/requirements.txt

FROM ci AS dev

# Define the env variables / args linked to the user
ENV USER=app
ARG HOST_UID
ARG HOST_GID
ARG HOME_DIRECTORY

# Ensure ARGS are provided
RUN test -n "$HOME_DIRECTORY" && test -n "$HOST_UID" && test -n "$HOST_GID"

# Create group and user: it allows to avoid permissions issue
RUN groupadd --force --gid $HOST_GID $USER && \
        useradd -M --home $HOME_DIRECTORY --base-dir $HOME_DIRECTORY \
        --uid $HOST_UID --gid $HOST_GID --shell "/bin/bash" $USER

# Create home directory + directory to store pip packages
RUN mkdir -p $HOME_DIRECTORY && mkdir -p $PYTHONUSERBASE
RUN chown -R $USER:$USER $PYTHONUSERBASE
RUN chown -R $USER:$USER $HOME_DIRECTORY

# Default user
USER $USER

# Add Python bin to PATH
ENV PATH=$PATH:$PYTHONUSERBASE/bin

# Copy all the files into HOME_DIRECTORY after installing the requirements
COPY . $HOME_DIRECTORY

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Install our pkg
RUN pip install --prefix $PYTHONUSERBASE -e .
