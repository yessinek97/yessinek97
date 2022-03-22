# This Makefile provides shortcut commands to facilitate local development.

# LAST_COMMIT returns the curent HEAD commit
LAST_COMMIT = $(shell git rev-parse --short HEAD)

# VERSION represents a clear statement of which tag based version of the repository you're actually running.
# If you run a tag based version, it returns the according HEAD tag, otherwise it returns:
# * `LAST_COMMIT-staging` if no tags exist
# * `BASED_TAG-SHORT_SHA_COMMIT-staging` if a previous tag exist
VERSION := $(shell git describe --exact-match --abbrev=0 --tags $(LAST_COMMIT) 2> /dev/null)
ifndef VERSION
	BASED_VERSION := $(shell git describe --abbrev=3 --tags $(git rev-list --tags --max-count=1))
	ifndef BASED_VERSION
	VERSION = $(LAST_COMMIT)-staging
	else
	VERSION = $(BASED_VERSION)-staging
	endif
endif

# Docker variables
DOCKER_HOME_DIRECTORY = /home/app

DOCKER_IMAGE_NAME = registry.gitlab.com/instadeep/biondeep-ig
DOCKER_IMAGE_TAG = $(VERSION)
DOCKER_IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
DOCKER_IMAGE_CI = $(DOCKER_IMAGE_NAME):$(CI_COMMIT_SHORT_SHA)

DOCKER_RUN_FLAGS = --rm --volume $(PWD):$(DOCKER_HOME_DIRECTORY)

DOCKERFILE = Dockerfile

# Build commands

.PHONY: build build-arm build-ci

define docker_buildx_template
	docker buildx build --platform=$(1) --progress=plain . \
		-f $(DOCKERFILE) -t $(2) --build-arg HOST_GID=$(shell id -g) \
		--build-arg HOST_UID=$(shell id -u) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY)
endef

define docker_build_ci_template
	docker build --target=ci --progress=plain . -f $(1) -t $(2)
endef

build:
	$(call docker_buildx_template,linux/amd64,$(DOCKER_IMAGE))

build-arm:
	$(call docker_buildx_template,linux/arm64,$(DOCKER_IMAGE))

build-ci:
	$(call docker_build_ci_template,$(DOCKERFILE),$(DOCKER_IMAGE_CI))
	docker tag $(DOCKER_IMAGE_CI) $(DOCKER_IMAGE)
# Push commands

.PHONY: push-ci

push-ci:
	docker push $(DOCKER_IMAGE)
	docker push $(DOCKER_IMAGE_CI)

# Dev commands

.PHONY: test bash docs

test: build
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) pytest --doctest-modules --verbose

bash: build
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) sh -c "pip install --user -e . && /bin/bash"

docs: build
	docker run $(DOCKER_RUN_FLAGS) -p 8000:8000 $(DOCKER_IMAGE) mkdocs serve

#IG Docker
IG_IMAGE_NAME=ig_train
IG_CONTAINER_NAME=ig_container
ig_build:
	docker build -t $(IG_IMAGE_NAME) --build-arg gid=$$(id -g)  --build-arg uid=$$(id -u)  -f Dockerfile.ig .

ig_run:
	docker run -it -d -e MACHINE_ID=`hostname`   --name $(IG_CONTAINER_NAME)  -v ${PWD}:/home/appuser/biondeep_ig  $(IG_IMAGE_NAME):latest
ig_bash:
	docker exec -it $(IG_CONTAINER_NAME) sh -c "pip install --user -e . && /bin/bash"
ig_rm:
	docker stop  $(IG_CONTAINER_NAME)
	docker rm  $(IG_CONTAINER_NAME)
