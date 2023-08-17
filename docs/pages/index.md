# BioNDeep IG SOW37

This website contains documentation for the [`biondeep-IG`](https://gitlab.com/instadeep/biondeep-ig) repository.

### IG
The IG project is the last step in the BionDeep pipeline. We develop an immunogenicity model that predicts the immunogenic potential of the mutant peptide (neoantigen), which is defined as its ability to trigger an immune response in the patient by provoking the generation-specific T-cells. This is an essential step in mutant peptide selection for stimulating a tumour-specific immune response in patients and thus has a significant therapeutic impact. To identify the determinant of T-cell response, immunogenicity models are trained on data from patients in which the immune response to candidate
peptides, in terms of CD8+ (for class I major histocompatibility complex molecules, MHC-I) or CD4+ (for class II
major histocompatibility complex molecules, MHC-II) levels, is monitored. For training, the model takes as input the
similarity between wild-type and mutant peptides and a large variety of information about both peptides, including the
binding and presentation probabilities predicted from previously developed models, biochemical features, positional and
sequence-based features, structural features of the pMHC complexes that have been simulated in-silico, etc.

See the [IG section](ig/installation.md) for more detail.
