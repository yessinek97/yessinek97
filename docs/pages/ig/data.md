# Data overview

The following datasets were provided by BioNTech are **confidential**. These datasets must not be downloaded or used on local machines. Please use the Paris DGX station or a GCP VM instance.

To manipulate (read and write) data from the GCP buckets, use the implemented [Push & Pull commands](push_pull.md).

## BionDeep Binding score

### Training/validation dataset : Public

The **cleaned** `public` dataset is available on GCP:

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Biondeep/data/Binding/public.clean.csv
```

### Test dataset : Optima

The **cleaned** `optima` dataset is available on GCP:

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Biondeep/data/Binding/optima.clean.csv
```

### Test dataset - Sahin et al.

The **cleaned** `sahin` dataset is available on GCP:

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Biondeep/data/Binding/sahin.clean.csv
```

### Features configuration

The file must be downloaded under the same folder as the data

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Biondeep/data/Binding/features_IG_16_11_2022_ensemble_binding_bnt_biondeep_pmhc.yml
```

## NetMHCpan score

### Training/validation dataset : Public

The **cleaned** `public` dataset is available on GCP:

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Netmhcpan/data/BaseFeaturesGo/public.clean.go.csv
```

### Test dataset : Optima

The **cleaned** `optima` dataset is available on GCP:

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Netmhcpan/data/BaseFeaturesGo/optima.clean.go.csv
```

### Test dataset - Sahin et al.

The **cleaned** `sahin` dataset is available on GCP:

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Netmhcpan/data/BaseFeaturesGo/sahin.go.csv
```

### Features configuration

The file must be downloaded under the same folder as the data

```bash
gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Netmhcpan/data/BaseFeaturesGo/features_IG_16_11_2022_Netmhcipan_bnt_netmhcpan_pmhc_go.yml
```
