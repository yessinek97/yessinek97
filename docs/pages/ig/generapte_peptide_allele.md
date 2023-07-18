To generate the peptide allele pairs for single or multiple datasets all you have to do is pass
the datasets as arguements along the (apeptide, allele) parir column names, all the datasets should
have the same column names.
This will generate the file in the specified location.

- In Docker container:
```bash
generate-pairs -d <data-path-1> -d <data-path-2> -p <peptide-column-name> -a <allele-column-name>  -o  <file-output-path>
```
- In Conda environment:
```bash
python -m ig.main generate-pairs -d <data-path-1> -d <data-path-2> -p <peptide-column-name> -a <allele-column-name>  -o  <file-output-path>
```

Example:

- In Docker container:
```bash
generate-pairs -d data/biondeep_optima.csv -d data/biondeep_sahin.csv -p tested_peptide_biondeep_mhci -a allele_biondeep_mhci  -o  data/peptide_allele_pair.csv
```
- In Conda environment:
```bash
python -m ig.main generate-pairs -d data/biondeep_optima.csv -d data/biondeep_sahin.csv -p tested_peptide_biondeep_mhci -a allele_biondeep_mhci  -o  data/peptide_allele_pair.csv
```

```bash
Options:
  -d       PATH    paths to datasets [required]

  -p       TEXT    peptide column name [required]

  -a       TEXT    allele column name  [required]

  -o       PATH    Path to the output data file [required]
```
