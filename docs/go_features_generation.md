# Anc2Vec package

To generate the go features using the Anc2vec package. First, you need to install the dependencies.

## Installation:

Run the following command:

```bash
conda create --name anc2vec python=3.8
conda activate anc2vec
pip3 install -U "anc2vec @ git+https://github.com/aedera/anc2vec.git"
pip3 install -r requirements.txt

```

## Gene ontology generation

If you want to generate and reduce the Go features embeddings then later include the reduced
embeddings, alongside the Go term CC RNA representations and finally merge them with specific
dataset use this command:

````bash
python biondeep_ig/gene_ontology_pipeline.py -data path_to_data -go 'go_term_mf go_term_bp go_term_cc'  -c 3 -t pca -o data/final_dataset_with_go_feat_and_go_cc_rna_rep.csv -e 'path of your Go embeddings' -s

where:
    -data: The input dataset path.
    -go: String separated go terms.
    -c: The number of components for Dimensionality Reduction.
    -t: The Dimensionality Reduction technique (pca,lsa,tsne).
    -o: The output path to save the dataset.
    -e: Path of ready-to-use Go terms embeddings.
    -s: A flag argument to save the Gene Ontology embeddings.

Example:
- To generate the Go embeddings run this command:

```bash

python biondeep_ig/gene_ontology_pipeline.py -data data/optima.clean.csv -go 'go_term_mf go_term_bp go_term_cc' -o data/final_dataset_optima_go_features_rna_representations.csv -c 3 -t pca -s

```

- To use your already generated Go embeddings run this command after specifying the embeddings path:

```bash
python biondeep_ig/gene_ontology_pipeline.py -data data/optima.clean.csv
-go 'go_term_mf go_term_bp go_term_cc' -o data/final_dataset_optima_go_features_rna_representations.csv -e data/go_embeddings_optima.clean.csv -c 3 -t pca -s

```
````
