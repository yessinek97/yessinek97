# Immunogenicity Data

## Raw Data (from BioNTech)

The raw data are stored inside
[google bucket](<https://console.cloud.google.com/storage/browser/biondeep-data/optima/Datasrets_11_10_2021;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=0&project=biontech-tcr&prefix=&forceOnObjectsSortingFiltering=false&pli=1>).

## To be updated

The files are stored under
`s3://biondeep-data/classi_data_for_instadeep/gnn/binding_predictor_downsampled_9mers/`

The IG data are used mainly to prepare features as follows:

1. TCR-based features (T-Cell Receptors).
2. Rosetta features.
3. Tf-idf Vectorizer.
4. Similarity of ProtBert Embeddings.

The Raw-Data for Rosetta-based features are constructed using (allele, mutated_peptide, wild_type
peptides, pMHC).

Using the script
[posegen](https://gitlab.com/instadeep/bioai-group/biondeep-structure/-/blob/main/gnn/gnn/data_gen/pmhc/generate_pmhc.py):

- We generate poses using (mut_peptide[pMHC]) to create mutated peptides set.
- We generate poses using (wt_peptide[pMHC]) to create wild type peptides set.
- We calculate energy-based features using Rosetta.

## Allele, Mutated_peptide, Wild_type_peptides, and pMHC

|                     | S3 Path                                                                             |
| ------------------- | ----------------------------------------------------------------------------------- |
| MUT_WT combinations | `s3://biondeep-data//biondeep-data/classi_data_for_instadeep/ig_dataset/mut_wt.csv` |

## Generated poses:

|                  | S3 Path                                                                    |
| ---------------- | -------------------------------------------------------------------------- |
| WT_peptides pdb  | `s3://biondeep-data/classi_data_for_instadeep/ig_dataset/wt_peptides/pdb`  |
| MUT_peptides pdb | `s3://biondeep-data/classi_data_for_instadeep/ig_dataset/mut_peptides/pdb` |

## Rosetta-Based generated features:

|                                      | S3 Path                                                                                         |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| WT_peptides Rosetta features         | `s3://biondeep-data/classi_data_for_instadeep/ig_dataset/wt_peptides/score`                     |
| WT_peptides Rosetta-features merged  | `s3://biondeep-data/classi_data_for_instadeep/ig_dataset/wt_peptides/rosetta_wt_peptides.csv`   |
| MUT_peptides Rosetta features        | `s3://biondeep-data/classi_data_for_instadeep/ig_dataset/mut_peptides/score`                    |
| MUT_peptides Rosetta-features merged | `s3://biondeep-data/classi_data_for_instadeep/ig_dataset/mut_peptides/rosetta_mut_peptides.csv` |

## Generated features

|                       | S3 Path                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------- |
| IG_features (Non-TCR) | `s3://biondeep-data//biondeep-data/classi_data_for_instadeep/ig_dataset/ig_non_tcr.csv` |
