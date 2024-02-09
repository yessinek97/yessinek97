# Go embeddings generation with Anc2Vec package

To generate the go features using the Anc2vec package. First, you need to install the dependencies.

## Installation

### Conda environment

- You can run the pipeline on a separate conda environment from the root directory which is **ig/**:

```bash
conda create --name anc2vec python=3.8
conda activate anc2vec
cd ig/
pip3 install -U "anc2vec @ git+https://github.com/aedera/anc2vec.git"
pip3 install -r requirements.txt
```

### Docker

- If you want to run this pipeline inside the docker container, use these commands to prepare the environment:

```bash
cd ig/
make build
make run
make bash
```

**PS**: You need to add the Google Storage [**Authentication credentials**](https://console.cloud.google.com/storage/browser/_details/biondeep-data/IG/biontech-tcr-16ca4aceba4c.json;tab=live_object?authuser=0) either on **Conda** or on **Docker** path to be able to read data from GCP buckets paths.

```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/client_secret.json'
```

## Gene ontology embeddings generation

The Gene Ontology pipeline helps generating gene ontology features following the mentioned steps below:

- Feature embedding generation from input sequences using **Anc2vec** library or using custom generated embeddings with another library.

- Dimensionality reduction for feature vectors to reduce the embeddings size and the number of outliers.

- The Go cc RNA localization features generation that give an idea about some certain gene ontology terms that refer to localization.

Use this command to run the gene ontology pipeline:

```bash
gene-ontology-pipeline -c gene ontology configuration file name -go 'string separated go term names' -o output save directory

where:
    -c: The gene ontology configuration file name.
    -go: String separated go terms.
    -o: The output path to save the embeddings/generated dataset.
```
### Gene ontology configuration file

This configuration file is used by the **Gene Ontology** pipeline to specify the settings for feature generation. It can be found at `configuration/gene_ontology.yml`. Here is the description of this configuration file:

```yaml
# Base datasets
dataset: # This section lists the used datasets to generate the needed features.
  dataset1: # This section describes dataset1.
    version: # This argument is used to specify the dataset version.
    # Example:
    version: "16_11_2022"
    paths: # This section lists the multiple data files included in dataset1.
      file1: # File1 path (It can be either local or GS path).
      file2: # File2 path (It can be either local or GS path).
    processing:
      trainable_features: # dataset1 features config file name
  dataset2: # This section describes dataset2.
    version: # This argument is used to specify the dataset version.
    # Example:
    version: "24_10_2022"
    paths: # This section lists the multiple data files included in dataset1.
      file1: # File1 path (It can be either local or GS path).
      file2: # File2 path (It can be either local or GS path).
    processing:
      trainable_features: # dataset2 features config file name
go_features: # This section describes the Ready-to-use gene ontology embeddings (cc,bp,mf)

  version: "16_11_2022" # This argument is used to specify the embeddings version.
  embedding_paths: # This section specifies the embedding paths (local, GS path)
    file1: # File1 embedding path (It can be either local or GS path).
    file2: # File2 embedding path (It can be either local or GS path).
    file3: # File3 embedding path (It can be either local or GS path).
    file4: # File4 embedding path (It can be either local or GS path).


  dimensionality_reduction: # This section describes the Dimensionality reduction settings to reduce embedding vectors.
    technique: # This argument defines the dimensionality reduction technique (pca,lda,lsa,tsne).
    n_components: # This argument specifies the number of components used to reduce the embeddings vectors shape.
  save_embeddings: # This boolean defines whether to save the embeddings independently or not.
```


### Gene ontology configuration file example

- You can use this configuration file to set the **dataset** specifications with multiple binders (Netmhcpan, Biondeep), their versioned data file paths and also the settings for **go_features** generation.

```yaml
# Base datasets
dataset:
  netmhcpan:
    version: "16_11_2022"
    paths:
      public: gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Netmhcpan/data/BaseFeatures/public.clean.csv
      optima: gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Netmhcpan/data/BaseFeatures/optima.clean.csv
    processing:
      trainable_features: gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Netmhcpan/data/BaseFeatures/features_IG_16_11_2022_Netmhcipan_bnt_netmhcpan_pmhc.yml
  biondeep:
    version: "16_11_2022"
    paths:
      public: gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Biondeep/data/Pres/PresBaseFeatures/public.clean.csv
      optima: gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Biondeep/data/Pres/PresBaseFeatures/optima.clean.csv
    processing:
      trainable_features: gs://biondeep-data/IG/BntPipelineData/IG_16_11_2022/Processing/Biondeep/data/Pres/PresBaseFeatures/features_IG_16_11_2022_ensemble_expression_presentation_bnt_biondeep_pmhc.yml
go_features:
  # Ready-to-use gene ontology embeddings (cc,bp,mf)
  version: "16_11_2022"
  embedding_paths:
    public: gs://biondeep-data/IG/InstaDeepPipelineData/GeneOntology/Anc2VecEmbeddings-PCA-3/IG_16_11_2022/public_go_embeddings_16_11_2022.csv
    optima:
      gs://biondeep-data/IG/InstaDeepPipelineData/GeneOntology
      /Anc2VecEmbeddings-PCA-3/IG_16_11_2022/optima_go_embeddings_16_11_2022.csv
    optima_pd:
      gs://biondeep-data/IG/InstaDeepPipelineData/GeneOntology
      /Anc2VecEmbeddings-PCA-3/IG_16_11_2022/optima_pd_go_embeddings_16_11_2022.csv
    sahin:
      gs://biondeep-data/IG/InstaDeepPipelineData/GeneOntology
      /Anc2VecEmbeddings-PCA-3/IG_16_11_2022/sahin_go_embeddings_16_11_2022.csv

  # Dimensionality reduction params to reduce embedding vectors
  dimensionality_reduction:
    technique: pca
    n_components: 3
  save_embeddings: True
```
- To generate embeddings from scratch using Anc2vec package, you can comment the key-value parameters under **embedding_paths**.

- To use your custom Go embeddings run this command after specifying the embedding paths under **embedding_paths** parameter under in **gene_ontology.yml**.

- You can then run this command to start the gene ontology pipeline:
```
gene-ontology-pipeline -c gene_ontology.yml -go 'go_term_mf go_term_bp go_term_cc' -o data/go_data_pipeline

```
### PS: Using local paths instead of Google storage paths is much faster.

- After the pipeline succeeds, a new folder called **go_data_pipeline** is going to be generated under which you can find two folders named **netmhcpan** and **biondeep** according to dataset binder names with 3 sub-folders (**configuration**, **embeddings** and **generated**) each.
They contain respectively the updated configuration file with the new generated features, the An2vec embeddings and the final generated datasets that include the reduced **gene ontology vector embeddings** to 3 components('go_term_mf_embed_vector_0', 'go_term_bp_embed_vector_0', 'go_term_cc_embed_vector_0', 'go_term_mf_embed_vector_1', 'go_term_bp_embed_vector_1', 'go_term_cc_embed_vector_1', 'go_term_mf_embed_vector_2', 'go_term_bp_embed_vector_2', 'go_term_cc_embed_vector_2') and also the **go_cc_rna_loc** features ("go_cc_rna_loc_membrane",
            "go_cc_rna_loc_nucleus",
            "go_cc_rna_loc_endoplasmic",
            "go_cc_rna_loc_ribosome",
            "go_cc_rna_loc_cytosol",
            "go_cc_rna_loc_exosome")
