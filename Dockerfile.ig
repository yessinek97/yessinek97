
FROM python:3.8-slim-buster

ARG uid
ARG gid
RUN apt-get update
RUN apt-get  install sudo libgomp1 git -y
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --quiet pip setuptools && pip install --no-cache-dir -r /tmp/requirements.txt && rm -rf /tmp/*
RUN groupadd -f -g $gid  appuser
RUN useradd -r -m -u $uid -g $gid -o -s /bin/bash appuser
USER appuser
WORKDIR   /home/appuser/biondeep_ig
# update path for pip install --user -e . later in exec
ENV PATH="/home/appuser/.local/bin:${PATH}"
#TODO add jupyter
