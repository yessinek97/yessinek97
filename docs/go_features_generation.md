# Anc2Vec package

To generate the go features using the Anc2vec package. First, you need to install the dependencies.

## Installation:

Run the following command:

```bash
conda create --name anc2vec python=3.6
conda activate anc2vec
pip3 install label-studio --ignore-installed certifi
pip3 install -U "anc2vec @ git+https://github.com/aedera/anc2vec.git"

```

## Gene ontology generation

If you want to generate and reduce the Go features embeddings then later include the reduced
embeddings, alongside the Go term CC RNA representations and finally merge them with specific
dataset use this command:

```bash
python biondeep_ig/gene_ontology_pipeline.py -data path_to_data -go 'go_term_mf go_term_bp go_term_cc'  -c 3 -t pca -o data/final_data_with_go_feat_and_go_cc_rna_rep.csv -e 'path of your Go embeddings'

where:
    -data: The input dataset path.
    -go: String separated go terms.
    -c: The number of components for Dimensionality Reduction.
    -t: The Dimensionality Reduction technique (pca,lsa,tsne).
    -o: The output path to save the dataset.
    -e: Path of ready-to-use Go terms embeddings.
    -s: A boolean to save embeddings only.

Example:
- To generat the Go embeddings run this command:

```

python biondeep_ig/gene_ontology_pipeline.py -data notebooks/data/optima.clean.csv -go 'go_term_mf
go_term_bp go_term_cc' -o notebooks/data/final_dataset_optima.csv -c 3 -t pca -s True

```

- To use your already generated Go embeddings run this command after specifying the embeddings path:

```

python biondeep_ig/gene_ontology_pipeline.py -data notebooks/data/optima.clean.csv -go 'go_term_mf
go_term_bp go_term_cc' -o notebooks/data/final_dataset_optima.csv -e
notebooks/data/final_dataset_optima_test.csv -c 3 -t pca -s True

```

```
