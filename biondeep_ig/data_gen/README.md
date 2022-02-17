# Bio IG data generation

This module contains the necessary code to generate pMHC and TCR-pMHC structures for datasets of
allele-peptide pairs.

## `ig`

The `ig.data_gen` module contains the scripts used in the data generation and also the scripts used
to prepare the dataset fed into the pipeline.

## `pipeline`

The `pipeline` module deals with running the data generation scripts on InstaDeep's infra and also
provides an alternative way to launch the pipeline on a single machine. For the former,
[Argo](https://argoproj.github.io/) is used to scale the process and distribute the task to multiple
nodes and executes the workflows defined in `pipeline.argo`. For the latter, we added scripts in
`pipeline.PosGeneration` (ending with `_parallel.sh`) to leverage the power of
[parallel](https://www.gnu.org/software/parallel/man.html) and parallelise the workload on multiple
cpus.
