IMAGE_NAME = ig-dev
CONTAINER_NAME = ig_dev
DOCKER_REGISTRY = registry.gitlab.com/instadeep/biondeep-ig
TAG = latest
TOKEN = not set
USERNAME = not set

DOCKER_IMAGE_LATEST = $(IMAGE_NAME):latest
HOME_DIRECTORY = /home/app/ig
DOCKER_RUN_FLAGS = -d --volume $(PWD):$(HOME_DIRECTORY) -e MACHINE_ID=`hostname` --name $(CONTAINER_NAME)


# Print help by default.
.DEFAULT_GOAL := help

.PHONY: help login build  bash

ifneq ($(TAG), latest)
	IMAGE_TAG = $(TAG)
else
	IMAGE_TAG = latest
endif

ifeq ($(PULL), true)
	PULL = $(PULL)
else
	PULL = false
endif

login:
ifneq ($(TOKEN), not set)
	ifneq (${USERNAME}, not set)
		docker login $(DOCKER_REGISTRY) -u ${USERNAME} -p ${TOKEN}
	endif
else
	docker login $(DOCKER_REGISTRY)
endif

build: ## Builds the docker image.
ifeq ($(PULL),true)
	docker pull $(DOCKER_REGISTRY):$(IMAGE_TAG)
	docker image tag $(DOCKER_REGISTRY):$(IMAGE_TAG) $(DOCKER_IMAGE_LATEST)
else
	docker build -t $(IMAGE_NAME)  --build-arg host_gid=$$(id -g) --build-arg host_uid=$$(id -u) -f Dockerfile.local .
endif

run: ## Create the container.
	docker run -it $(DOCKER_RUN_FLAGS)  $(DOCKER_IMAGE_LATEST)

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
