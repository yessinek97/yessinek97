# Push pull command

## Google Cloud Bucket

We have two buckets:

biondeep-data: store datasets and metrics

biondeep-models: store models

In order to have access to the buckets you have to ask a developer to give you the file with the
credentials. This file must be named biontech-tcr-16ca4aceba4c.json and it must be put at the root
of the project.

## Share models / data thanks to Google Buckets

There are 3 command lines to share data and models:

push: push some data/model to the biondeep-data/biondeep-models Google Bucket

pull: pull some data/model from the biondeep-data/biondeep-models Google Bucket to your local
machine

Note: Please make sure to export the shared json file each time you run the command:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=CREDENTIALS_FILE.json

```

### The push command:

- In Docker container:
```bash
push --local_path ig/ig/models/model_trained/ --bucket_path gs://biondeep-models/IG/Folder/
```
- In Conda environment:
```bash
python -m ig.main compute-comparison-score -d <data_path>  -l <label_name>  -c  <column_name>
```

For example:

- In Docker container:
```bash
push --local_path ig/ig/models/netmhc_sahin_public_19_05_2022/ --bucket_path gs://biondeep-models/IG/netmhc_sahin_public_19_05_2022/
```
- In Conda environment:
```bash
python -m ig.main compute-comparison-score -d <data_path>  -l <label_name>  -c  <column_name>
```

### The pull command

- In Docker container:
```bash
pull --bucket_path gs://biondeep-models/IG/netmhc_sahin_public_19_05_2022/ --local_path /ig/model_pulled/
```
- In Conda environment:
```bash
python -m ig.main pull --bucket_path gs://biondeep-models/IG/netmhc_sahin_public_19_05_2022/ --local_path /ig/model_pulled/
```

For example:

- In Docker container:
```bash
pull --bucket_path gs://biondeep-models/IG/netmhc_sahin_public_19_05_2022/ --local_path /ig/local_netmhc_sahin_public_19_05_2022/
```
- In Conda environment:
```bash
python -m ig.main pull --bucket_path gs://biondeep-models/IG/netmhc_sahin_public_19_05_2022/ --local_path /ig/local_netmhc_sahin_public_19_05_2022/
```

14:45 python -m ig.main pull --bucket_path gs://biondeep-models/IG/nethmcpan_sahin_public
--local_path /raid/amels/ig/model_pull_test/
