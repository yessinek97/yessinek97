IMAGE_NAME = ig-dev
CONTAINER_NAME = ig_dev

DOCKER_REGISTRY = registry.gitlab.com/instadeep
PROJECT_NAME = biondeep-ig

TAG = latest

HOME_DIRECTORY = /home/app/ig
DOCKER_RUN_FLAGS = --gpus all -d --volume $(PWD):$(HOME_DIRECTORY) -e MACHINE_ID=`hostname` --name $(CONTAINER_NAME)

#Variables for creating the biondeepIG Container and documentation Docker container
DOCKER_IMAGE_TAG = $(shell git rev-parse --short HEAD)

DOCKER_IMAGE_NAME = $(DOCKER_REGISTRY)/$(PROJECT_NAME)
DOCKER_IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
DOCKER_IMAGE_LATEST = $(DOCKER_IMAGE_NAME):latest

DOCKER_IMAGE_NAME_DOCS = $(DOCKER_REGISTRY)/$(PROJECT_NAME)/docs
DOCKER_IMAGE_DOCS_LATEST = $(DOCKER_IMAGE_NAME_DOCS):latest

DOCKER_RUN_FLAGS_DOCS = -it --rm --volume $(PWD):$(HOME_DIRECTORY) -p $(MKDOCS_PORT):$(MKDOCS_PORT)
MKDOCS_PORT = 8000
RUN_MKDOCS = mkdocs serve -a 0.0.0.0:$(MKDOCS_PORT)

# Print help by default.
.DEFAULT_GOAL := help

.PHONY: help login build  bash

login:	##Login to the GitLab Docker registry
	docker login $(DOCKER_REGISTRY)

build: login ## Builds the docker image.
	docker pull $(DOCKER_IMAGE_LATEST)
	docker build -t $(DOCKER_IMAGE)  --build-arg TAG=$(TAG) \
									--build-arg host_gid=$$(id -g) \
									--build-arg host_uid=$$(id -u) \
									-f Dockerfile.local .

run: build ## Create the container.
	docker run -it $(DOCKER_RUN_FLAGS)  $(DOCKER_IMAGE)

bash: ## gets a bash in the container
	docker start  $(CONTAINER_NAME)
	docker exec -it $(CONTAINER_NAME) sh -c "pip install --user -e . && /bin/bash"


setup: ## Builds the image and gets a bash in the container.
	make build run
rm:
	docker stop  $(CONTAINER_NAME)
	docker rm  $(CONTAINER_NAME)
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


#Creating a Docker container for visualizing the documentation using mkdocs on local machine
build-docs:  ## Building the documentation docker container
	docker build -t $(DOCKER_IMAGE_DOCS_LATEST) --build-arg HOME_DIRECTORY=$(HOME_DIRECTORY) \
										-f Dockerfile.docs .

docs : build-docs #Running the documentation docker container inorder to visualize the documentation using mkdocs on a local machine (http://127.0.0.1:8000/)
	docker run $(DOCKER_RUN_FLAGS_DOCS) $(DOCKER_IMAGE_DOCS_LATEST) $(RUN_MKDOCS)
