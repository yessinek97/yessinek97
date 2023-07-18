## IG Data Generation (TCR-pMHC)

![Data IG generation pipeline](./static/data_ig_pipeline.png "Data IG generation pipeline")

The `ig.ig.data_gen` module takes care of generating pMHC and TCR-pMHC structures
for a given (**allele**, **peptide**) pair. The pipeline is divided into 3 step:

1. Dataset preparation: 2 CSV files are created for the pMHC and TCR-pMHC generation stages.
2. pMHC generation: for each (**allele**, **peptide**, **template**), a redocked+relaxed pMHC
   structure is generated.
3. pMHC generation: for each (**generated_pMHC**, **TCR**, **template**), N redocked+relaxed
   TCR-pMHC structure is generated (typically, N=4).

## Step 0: Setup Docker container

To run the pipeline, you'll need to build the Docker image and create a container from it.

```bash
# At the root of the repo, build the Docker image
make build
```

## Step 1: Dataset preparation

---

Note: Please be mindful that the pipeline will be run within the Docker image. When choosing the
path for the artefacts below, be sure that it exists and that your data will be stored in that
directory. For instance, if the `artefacts-dir` is `/home/app/artefacts`, you can copy your local
pMHC folder into the container:

```bash
# For pMHC templates
docker cp /path/to/pmhc/dir/. <CONTAINER_ID>:/home/app/artefacts/pmhc
# For TCR structures
docker cp /path/to/tcr/dir/. <CONTAINER_ID>:/home/app/artefacts/tcr
# For TCR-pMHC templates
docker cp /path/to/template/dir/. <CONTAINER_ID>:/home/app/artefacts/template
```

---

```bash
python3 ig/data_gen/ig/data_gen/generate_dataset.py \
    --datasets /home/app/data/public_ig.csv /home/app/data/optima_ig.csv \
    --pmhc-dir /home/app/data/pmhcs/ \
    --num-flags 2 \
    --out-pmhc-data-path /home/app/datasets/pmhc_data.csv \
    --tm-alignment-path /home/app/artefacts/tcrs_alignments.csv \
    --artefacts-dir /home/app/artefacts \
    --out-tcr-pmhc-data-path /home/app/datasets/tcr_pmhc_data.csv
```

All the options are shown here:

- datasets: list of paths to the raw input datasets.
- pmhc-dir: directory with the pMHC templates with format: `<ALLELE>.model00_0001.pdb`.
- num-flags: number of generated pMHCs for each (**allele**, **pair**) pair.
- out-pmhc-data-path: path to output pMHC input dataset for pipeline

- artefacts-dir: path to the artefacts directory (which must contains the 3 folders `pmhc`, `tcr`
  and `template` containing the pMHC templates, the TCR structures and the TCR-pMHC templates
  respectively).

- tm-alignment-path: path to alignment CSV file (TM score between every possible (TCR, template)
  pair.
- out-tcr-pmhc-data-path: path to output TCR-pMHC input dataset for pipeline.

The output CSVs will be used as inputs for the pipeline. The artefacts are used during data
generation, e.g `<artefacts-dir>/pmhc/A2001.model00_0001.pdb` will be used as template to generate
the (A2001, AIMPSBCUA) pair.

## Step 2: pMHC generation

To generate the pMHC structures, run the convenient
`ig.data_gen.pipeline.PosGeneration.gen_pmhc_poses_parallel.sh` script. To do so:

0. Open a [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) session (this will prevent the
   termial from disconnecting if the machine get into sleep mode, or if you're using a VM and want
   to close your IDE/terminal without stopping the execution of the code):

```bash
# If not installed
sudo apt install tmux # for Linux
brew install tmux # for MacOS

# Create session
tmux new -s pmhc_gen
```

1. Run the Docker image and bash into it:

```bash
# Create a tmux session
# At the root of the repo
make bash
```

2. Run the generation script:

```bash
# Once in the container
chmod +x ./ig/data_gen/pipeline/PosGeneration/gen_pmhc_poses_parallel.sh
./ig/data_gen/pipeline/PosGeneration/gen_pmhc_poses_parallel.sh -i /path/to/pmhc_data.csv -o /home/app/generated_pmhc
```

All the pMHC structures will be generated in `/home/app/generated_pmhc` folder. Note that this
folder must have the same parent directory as the artefacts folder, e.g `/home/app/artefacts`.

## Step 3: TCR-pMHC generation

Once the pMHCs are generated, we can launch the TCR-pMHC generation.

---

Note: Make sure to have your TCR and TCR-pMHC structures in the `/home/app/artefacts` directory as
described above.

---

0. Open a new tmux session:

```bash
tmux new -s tcr_pmhc_gen
```

1. Run the Docker image and bash into it:

```bash
# Create a tmux session
# At the root of the repo
make bash
```

2. Run the generation script:

```bash
# Once in the container
chmod +x ./ig/data_gen/pipeline/PosGeneration/gen_tcr_pmhc_poses_parallel.sh
./ig/data_gen/pipeline/PosGeneration/gen_tcr_pmhc_poses_parallel.sh -i /path/to/tcr_pmhc_data.csv -o /home/app/generated_tcr_pmhc -n 4
```

All the TCR-pMHC structures will be generated in `/home/app/generated_tcr_pmhc` folder.

## Step 4: Merging TCR-pMHC structures into existing datasets.

Once the TCR-pMHC structures are generated, they can be merged into existing datasets used for
training or testing using the following command:

- In Docker container:
```bash
merge_tcr_pmhc -i <path_to_existing_dataset> -t <path_to_tcr_pmhc_structures> -o <output_directory> -c <configuration_file>
```
- In Conda environment:
```bash
python -m ig.main merge_tcr_pmhc -i <path_to_existing_dataset> -t <path_to_tcr_pmhc_structures> -o <output_directory> -c <configuration_file>
```

A default configuration file is available at
`ig/data_gen/ig/data_gen/config_merge_tcrpmhc.yml`. The mandatory attributes and
corresponding example values are as follows:

```yaml
binding_energy_column: "dg_separated" # Column name corresponding to the binding energy score.
right_on: # List of columns name from the tcr-pMHC data on which the merge is performed.
  - "peptide"
  - "allele"
left_on: # List of columns name the target dataset on which the merge is performed.
  - "tested_peptide_biondeep_mhci"
  - "tested_allele_biondeep_mhci"
select_on:
  "min" # Statistics applied to binding energy use to select and aggregate features.
  # Can be one of:
  # - "mean"
  # - "min"
  # - float in ]0;1[
```
