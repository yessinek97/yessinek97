# Generate pMHC structure

![pMHC pipeline](pmhc_gen.png "pMHC pipeline")

The `generate_pmhc.py` script generates a pMHC structure from a pMHC template and a target peptide
sequence. The procedure executes the following steps:

1. The template is loaded as a pyrosetta pose and its peptide sequence length is compared to the
   target peptide.
2. If the target peptide is longer, extra residues are inserted in the peptide structure, starting
   from a offset position (we don't allow too much perturbation at the start and end of the
   peptide).
3. The peptide structure is then mutated according to the target peptide sequence and each mutation
   step is followed by a minimization step.
4. A docking and a relaxation are then applied on the resulting complex.

Example command:

```bash
pmhc_gen --peptide MALMALMAL --init_pdb A0101_ILDFGLAKL.pdb --flag 1 --output_dir out/pmhc
```

Note: The output naming convention is defined as follows:

`{INIT_PDB_STEM}_{PEPTIDE_SEQUENCE}_{FLAG}_{min,relaxed}.pdb.gz`
