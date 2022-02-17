# TCR-pMHC Structure Generation

![TCR-pMHC pipeline](tcr_pmhc_gen.png "TCR-pMHC pipeline")

To generate TCR-pMHC structures, we should have the following data in advance

- a pMHC structure corresponding to (allele, peptide).
- a TCR structure.
- a TCR-pMHC structure template.

The three scripts tcr_pmhc_align.py, tcr_pmhc_extract.py and generate_tcr_pmhc.sh are used in a
hierarchical manner in the Argo workflow pipeline. First, tcr_pmhc_extract.py is called to extract
individual chains. Second, the tcr_pmhc_align.py is used to apply the rotation and transformation
computed by TMAlign. Finally, the generate_tcr_pmhc.sh, which uses the 2 former scripts as
intermediary steps, generates the target pMHC-TCR structure.

## `generate_tcr_pmhc.sh` (Requires TMAlign - use `biondeep_ig.Dockerfile`)

This is the main script that takes as input a pMHC PDB, a TCR PDB and a pMHC-TCR template PDB. It
aligns the former 2 PDBs with respect to their counterparts in the template using TMalign, then
"merges" the 2 aligned resulting PDBs into one single structure on which a final docking step is
applied using Rosetta. Scores are finally computed using Rosetta's InterfaceAnalyzer.

Example command:

```bash
chmod +x ./biondeep_ig/bio_ig_gen/ig/data_gen/tcr_pmhc/generate_tcr_pmhc.sh
./biondeep_ig/bio_ig_gen/ig/data_gen/tcr_pmhc/generate_tcr_pmhc.sh  -p /home/app/data/pmhcs/FLVQNIHTL_1_relax.pdb -t /home/app/data/tcrs/tcr-CAVPLYNNNDMRF-CASSDRGLGYGYTF.pdb -m /home/app/data/tcr_pmhc_templates/4JFF.rechained.pdb -o /home/app/data/out/rosetta -n 2
```

## `tcr_pmhc_align.py`

This script parses the outputs (TM-score and rotation/translation matrix) from TMalign and uses
PyRosetta to rotate and translate the pMHC and TCR structures.

```bash
tcr_pmhc_align --pmhc_pdb /home/app/data/pmhcs/FLVQNIHTL_1_relax.pdb --pmhc_tmalign /home/app/data/out/pmhc_peptide_align.txt --pmhc_tmalign_out /home/app/data/out/pmhc_peptide_align_out.txt \
--tcr_pdb /home/app/data/tcrs/tcr-CAVPLYNNNDMRF-CASSDRGLGYGYTF.pdb --tcr_tmalign /home/app/data/out/tcr_beta_align.txt  --tcr_tmalign_out /home/app/data/out/tcr_beta_align_out.txt \
--out_pdb /home/app/data/out/tcr_pmhc_aligned.pdb
```

## `tcr_pmhc_extract.py`

This script extracts the peptide structure from a pMHC PDB and the beta structure from a TCR PDB.

Example commands:

```bash
tcr_pmhc_extract --tcr /home/app/data/tcrs/tcr-CAVPLYNNNDMRF-CASSDRGLGYGYTF.pdb --pmhc /home/app/data/pmhcs/FLVQNIHTL_1_relax.pdb --template /home/app/data/tcr_pmhc_templates/4JFF.rechained.pdb --output_dir /home/app/data/out
```
