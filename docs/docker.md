## Using Docker as conda alternative

You can use Docker for training the models or inference on a separate dataset. There are three
stages for this pipeline to be functional.

First you need to build the docker image:

```
# Build Docker Image
make ig_build

```

Then you need to run the built image to create a separate container:

```
# run the image
make ig_run

```

finlay, you need to run the bash of the created container:

```
# Make script executable
make ig_bash

```
