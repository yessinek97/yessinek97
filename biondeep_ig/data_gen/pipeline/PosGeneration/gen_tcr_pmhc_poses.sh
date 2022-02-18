#!/bin/bash

# arguments
while getopts p:r:t:o:n: param
do
    case "${param}" in
        p) PMHC=${OPTARG};;
        r) TCR=${OPTARG};;
        t) TEMPLATE=${OPTARG};;
        o) BASE_OUT_DIR=${OPTARG};;
        n) NUM_DOCKING=${OPTARG};;
        *) echo "Unknown flag." 1>&2 && exit 1
    esac
done


# check arguments
if [[ -z "$NUM_DOCKING" ]]; then
    echo "Must define docking number for operation." 1>&2
    exit 1
fi
if [[ -z "$PMHC" ]]; then
    echo "Must define path of pMHC pdb." 1>&2
    exit 1
fi
if [[ -z "$TCR" ]]; then
    echo "Must define path of TCR pdb." 1>&2
    exit 1
fi
if [[ -z "$TEMPLATE" ]]; then
    echo "Must define path of the template pdb." 1>&2
    exit 1
fi
if [[ -z "$BASE_OUT_DIR" ]]; then
    echo "Must define base output dir." 1>&2
    exit 1
fi

# create new directory for pMHC/TCR/Template combination
PMHC_STEM=$(echo "$PMHC" | rev | cut -d'/' -f -1 | rev)
PMHC_STEM=${PMHC_STEM//".gz"/""}
PMHC_STEM=${PMHC_STEM//".pdb"/""}

TCR_STEM=$(echo "$TCR" | rev | cut -d'/' -f -1 | rev)
TCR_STEM=${TCR_STEM//".pdb"/""}

TEMPLATE_STEM=$(echo "$TEMPLATE" | rev | cut -d'/' -f -1 | rev)
TEMPLATE_STEM=${TEMPLATE_STEM//".pdb"/""}

OUT_DIR=$BASE_OUT_DIR/${PMHC_STEM}_${TCR_STEM}_${TEMPLATE_STEM}
mkdir -p $OUT_DIR

echo "PROCESSING pMHC: ${PMHC_STEM} - TCR: ${TCR_STEM} - TEMPLATE: ${TEMPLATE_STEM}..."

chmod +x ./biondeep_ig/data_gen/ig/data_gen/tcr_pmhc/generate_tcr_pmhc.sh
./biondeep_ig/data_gen/ig/data_gen/tcr_pmhc/generate_tcr_pmhc.sh -p $PMHC -t $TCR -m $TEMPLATE -n $NUM_DOCKING -o $OUT_DIR

NUM_FILES_GENERATED=$(ls ${OUT_DIR} | wc -l)
echo "${NUM_FILES_GENERATED} were generated in ${OUT_DIR}"

# normal exit
exit 0
