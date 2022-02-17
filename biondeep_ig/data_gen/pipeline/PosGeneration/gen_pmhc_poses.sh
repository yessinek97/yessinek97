#!/bin/bash

# arguments
while getopts r:i:a:p:o: param
do
    case "${param}" in
        r) FLAG=${OPTARG};;
        i) INIT_PDB=${OPTARG};;
        a) ALLELE=${OPTARG};;
        p) PEPTIDE=${OPTARG};;
        o) OUTPUT_DIR=${OPTARG};;
        *) echo "Unknown flag." 1>&2 && exit 1
    esac
done


# check arguments
if [[ -z "$FLAG" ]]; then
    echo "Must define flag for operation." 1>&2
    exit 1
fi
if [[ -z "$INIT_PDB" ]]; then
    echo "Must define path of allele init pdb." 1>&2
    exit 1
fi
if [[ -z "$ALLELE" ]]; then
    echo "Must define the input Allele." 1>&2
    exit 1
fi
if [[ -z "$PEPTIDE" ]]; then
    echo "Must define the input peptide." 1>&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Must define the output directory." 1>&2
    exit 1
fi

# File paths must match the outputs from pmhc_gen
PDB_STEM=$(echo "$INIT_PDB" | rev | cut -d'/' -f -1 | rev)
PDB_STEM=${PDB_STEM//".pdb"/""}
OUT_FILE_MIN=${OUTPUT_DIR}/${PDB_STEM}_${PEPTIDE}_${FLAG}_min.pdb
OUT_FILE_RELAX=${OUTPUT_DIR}/${PDB_STEM}_${PEPTIDE}_${FLAG}_relax.pdb

if [ -f "$OUT_FILE_RELAX.gz" ]; then
    echo "$OUT_FILE_RELAX.gz already generated!"
    exit 0
fi

echo "PROCESSING - allele: ${ALLELE} - peptide: ${PEPTIDE} - FLAG: ${FLAG}"

pmhc_gen --peptide $PEPTIDE --init_pdb $INIT_PDB --flag $FLAG --output_dir ${OUTPUT_DIR}

# Analyze interface and store scores in files.
if [[ -f "$OUT_FILE_MIN" ]]; then
    /Rosetta/main/source/bin/InterfaceAnalyzer.default.linuxgccrelease -pack_separated -pack_input -out:level 10 -in:file:s $OUT_FILE_MIN -out:file:score_only ${OUTPUT_DIR}/ia_${PDB_STEM}_${PEPTIDE}_${FLAG}_min.sc -interface A_C
    gzip $OUT_FILE_MIN
fi
if [[ -f "$OUT_FILE_RELAX" ]]; then
    /Rosetta/main/source/bin/InterfaceAnalyzer.default.linuxgccrelease -pack_separated -pack_input -out:level 10 -in:file:s $OUT_FILE_RELAX -out:file:score_only ${OUTPUT_DIR}/ia_${PDB_STEM}_${PEPTIDE}_${FLAG}_relax.sc -interface A_C
    gzip $OUT_FILE_RELAX
fi

exit 0
