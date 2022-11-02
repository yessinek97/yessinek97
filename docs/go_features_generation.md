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

If you want to include the go features in a specific dataset:

```bash
python biondeep_ig/generate_go_anc2vec.py -data path_to_data -go 'go_term_mf go_term_bp go_term_cc' -o data/new_data_with_go_feat.csv

Examle:
python biondeep_ig/generate_go_anc2vec.py -data 'data/BioNDeep_transformer_publicMUT_20220601_BioNDeep_transformer_v2_struc.tsv' -go 'go_term_mf go_term_bp go_term_cc' -o data/new_data_with_go_feat.csv

```
