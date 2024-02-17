# GCP Push & Pull command

## Share data & models results thanks to Google Cloud Bucket

We have two buckets:

* [biondeep-data-IG](https://console.cloud.google.com/storage/browser/biondeep-data/IG/data) : for datasets

* [biondeep-models-IG](https://console.cloud.google.com/storage/browser/biondeep-data/IG/experiments): for models and metrics results

>🚨 In order to be able to Read & Write [DATA](https://console.cloud.google.com/storage/browser/biondeep-data/IG/data) and [Models](https://console.cloud.google.com/storage/browser/biondeep-data/IG/experiments) from the BioNdeep-IG GCP storage Buckets you must first do the [Google Storage authentication step](installation.md#google-storage-authentication)

There are 2 command lines to share and download data and models:

* `push`: to save some data/model to the biondeep-data/biondeep-models Google Bucket

* `pull`: to download some data/model from the biondeep-data/biondeep-models Google Bucket to your local machine

## In Docker container

### Docker push command

```bash
push --local_path /home/app/ig/models/model_Folder/ --bucket_path gs://biondeep-models/IG/model_Folder/
```

For example:

```bash
push --local_path /home/app/ig/models/netmhc_sahin_public_19_05_2022/ --bucket_path gs://biondeep-models/IG/netmhc_sahin_public_19_05_2022
```

### Docker pull command

```bash
pull --bucket_path gs://biondeep-models/IG/model_folder/ --local_path /home/app/ig/models/pulled_model/
```

For example:

```bash
pull --bucket_path gs://biondeep-models/IG/IG_last_version/ --local_path /home/app/ig/models/pulled_IG_last_version/
```

## In Conda environment

### Conda push command

```bash
python -m ig.main push --local_path PATH_To/biondeep-ig/models/model_Folder/ --bucket_path gs://biondeep-models/IG/model_Folder/
```

For example:

```bash
python -m ig.main push --local_path PATH_To/biondeep-ig/models/netmhc_sahin_public_19_05_2022/ --bucket_path gs://biondeep-models/IG/netmhc_sahin_public_19_05_2022/
```

### Conda pull command

```bash
python -m ig.main pull --bucket_path gs://biondeep-models/IG/model_Folder/ --local_path PATH_To/biondeep-ig/models/pulled_model/
```

For example:

```bash
python -m ig.main pull --bucket_path gs://biondeep-models/IG/IG_last_version/ --local_path PATH_To/biondeep-ig/models/pulled_IG_last_version/
```
