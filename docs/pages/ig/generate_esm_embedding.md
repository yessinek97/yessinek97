Generation of dot product ESM embeddings from Meta for a specific dataset is the following:

- In Docker container:
```bash
compute-embeddings  -d <data file> -cm <mutated column> -cw <wild type column> -o <output file>
```
- In Conda environment:
```bash
python -m ig.main compute-embeddings  -d <data file> -cm <mutated column> -cw <wild type column> -o <output file>

```

Example:

- In Docker container:
```bash
compute-embeddings  -d 'data.csv' -cm 'tested_peptide_biondeep_mhci' -cw 'wildtype_peptide_biondeep_mhci' -o output.csv
```
- In Conda environment:
```bash
python -m ig.main compute-embeddings  -d 'data.csv' -cm 'tested_peptide_biondeep_mhci' -cw 'wildtype_peptide_biondeep_mhci' -o output.csv
```

```bash
Options:
  -d       TEXT    CSV data [required]

  -cm      TEXT    mutated column [required]

  -cw      TEXT    wild type column  [required]

  -o       TEXT    output file [required]
```

The output file will contain the new column: dot product of the esm mutation and wild peptides embeddings.
