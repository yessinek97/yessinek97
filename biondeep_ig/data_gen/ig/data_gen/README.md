# Generate pMHC and TCR-pMHC structures

We set up a Rosetta-based pipeline to generate pMHC and TCR-pMHC structures for a set of given
(allele, peptide) tuples. The data preparation step is performed by the `generate_dataset.py` file.

For each combination:

- the first step consists in generating first a pMHC structure using a pMHC template: the latter is
  mutated to account for the desired peptide, then the structure is relaxed before going through a
  docking process.
- the second step uses the generated pMHC structure, a set of TCR structures and a TCR-pMHC
  template: the pMHC PDB is aligned w.r.t to the templates using the peptide as pivot, the same is
  done for the TCR PDB w.r.t to the templates using the beta chain as pivot. Then, the 2 aligned
  poses (pMHC + TCR) are merged into one single pose which finally gets redocked.

## Data preparation

First, before being able to generate structures, we need to select the desired (allele, peptide)
tuples, and the TCRs and template structure used in the pipeline. We chose to represent the data for
each step as CSV files: the first one contains for each row the allele, peptide and pMHC template.
The second CSV contains for each row the generated pMHC, the TCR and the TCR-pMHC template.
Regarding the latter, as we try several TCR for each pMHC (typically 20), several rows contain the
same pMHC but different TCR and template structures to be tested.

The whole data preparation process outputs 2 CSV files that are to be sent to the S3 output. To
generate them, simply call the `generate_dataset.py` as shown in the following example:

```
python3 generate_dataset.py \
    -d /home/app/datasets/public_ig.csv /home/app/datasets/optima_ig.csv \
    -p /home/app/datasets/pmhcs/ \
    -n 2
    --tm-alignment-path /home/app/datasets/tcrs_alignments.csv
```

where:

- `-d `is the list of paths to the selected dataset CSVs.
- `-p` is the path to the folder containing the available pMHC structures (for step 1 of the
  pipeline).
- `-n` is the number of flags (number of output pMHC structures).
- `--tm-alignment-path` is the path to the CSV containing the alignment score between the available
  TCRs and TCR-pMHC templates.

Additional arguments exist, their descriptions can be found in the Python script.

## TCR-pMHC structure generation

The pipeline can be decomposed into 3 steps, including an optional one:

- the pMHC generation script can be found in the `pmhc` folder. Please read the `pmhc.README.md`
  file for more information.
- the TCR generation script is not called in the pipeline as the TCR structures are supposed to be
  already created and stored in the S3 bucket. However, the `tcr.tcrmodel.sh` script can be called
  to generate a TCR structure using Rosetta built-in program, given the sequences for the CDR3a and
  CDR3b sequences.
- the TCR-pMHC generation script can be found in the `tcr_pmhc` folder. Please read the
  `tcr_pmhc.README.md` file for more information.
