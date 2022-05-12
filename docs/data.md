## Datasets

The following datasets have been provided by BioNTech are **confidiential**. These datasets must not
be downloaded and used on local machines. Please use the Paris DGX station or a GCP VM instance.

### Training/validation dataset : Public

The **cleaned** `public` dataset is available on GCP:

```bash
gs://biondeep-data/IG/10_03_2022/public_2022_03_10:clean:latest.csv
```

### Test dataset : Optima

The **cleaned** `optima` dataset is available on GCP:

```bash
gs://biondeep-data/IG/10_03_2022/optima_2022_03_10:latest:floats.csv
```

### Test dataset - Sahin et al.

The **cleaned** `sahin` dataset is available on GCP:

```bash
gs://biondeep-data/IG/10_03_2022/sahin_clean_2022_04_06.csv
```

The **raw** version of this dataset is found on BioNTech sftp server:

```bash
/biontech-de-sftp-bucket/BNTpub/ImmuneResponse_20220310/sahin_20211121_out_TCGA_updated_pres.tsv
```

Connection to this sftp server is **only** possible via the Paris DGX station using the command:

```bash
sftp -i ~/.ssh//bnt_rsa biontech-de@s-b2c3dbdb082e4f01b.server.transfer.us-east-1.amazonaws.com
```
