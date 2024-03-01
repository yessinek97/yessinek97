# To replace with the following image once it is available on the registry
FROM python:3.8-slim-buster

# Update and upgrade your base image
RUN apt-get update && \
        apt-get upgrade -y
# Install required system dependencies and clear cache
RUN apt-get  install sudo libgomp1 -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install git parallel -y && \
        apt-get clean

# Copy the requirements file into /tmp directory
COPY ./requirements.txt /tmp/requirements.txt


# Install python requirements
RUN pip install --upgrade --quiet pip setuptools && \
        pip install --no-cache-dir -r /tmp/requirements.txt && \
        rm -rf /tmp/*
