#!/bin/bash
# exit if any command fails
set -e

# arguments
while getopts p:t:m:o:n: param
do
    case "${param}" in
        p) PMHC=${OPTARG};;
        t) TCR=${OPTARG};;
        m) TEMPLATE=${OPTARG};;
        o) OUT_DIR=${OPTARG};;
        n) NUM_DOCKING=${OPTARG};;
        *) echo "Unknown flag." 1>&2 && exit 1
    esac
done

# Check if inputs have already been processed
score_path=${OUT_DIR}/ia_dock.sc
if [ -f "$score_path" ]; then
    echo "$score_path already generated..."
    exit 0
fi

# Check if input pMHC exists
if [ ! -f "$PMHC" ]; then
    if [ ! -f "${PMHC::-3}" ]; then
        echo "$PMHC does not exist. It was probably not created during pMHC generation..." 1>&2
        exit 0
    fi
fi

# Uncompress pMHC file if needed
if [[ $PMHC == *.pdb.gz ]]; then
    PMHC=${PMHC::-3}
    if ! (test -f "$PMHC"); then
        echo "Uncompressing pMHC file..."
        gzip -d ${PMHC}.gz
    fi
fi

# Extract chains from pMHC, TCR and from pMHC-TCR complex template:
#   - Peptide from pMHC complex
#   - TCR beta chain from TCR complex
echo "Extracting chains from complexes..."
tcr_pmhc_extract --tcr $TCR --pmhc $PMHC --template $TEMPLATE --output_dir $OUT_DIR

# Check the chains have been extracted
ls -lh $OUT_DIR

# align chains
# https://zhanglab.ccmb.med.umich.edu/TM-align/
# file names need to be consistent with tcr_pmhc_extract
pep_path=${OUT_DIR}/pmhc_peptide.pdb                # Extracted Peptide PDB path
tpl_pep_path=${OUT_DIR}/template_pmhc_peptide.pdb   # Template Peptide PDB path
pep_align=${OUT_DIR}/pmhc_peptide_align.txt         # TMalign Peptide rotation file path
pep_align_out=${OUT_DIR}/pmhc_peptide_align_out.txt # TMalign Peptide console output
tcr_beta_path=${OUT_DIR}/tcr_beta.pdb               # Extracted TCR beta chain PDB path
tpl_tcr_beta_path=${OUT_DIR}/template_tcr_beta.pdb  # Extracted Templated TCR beta chain PDB path
tcr_align=${OUT_DIR}/tcr_beta_align.txt             # TMalign TCR beta chain rotation file path
tcr_align_out=${OUT_DIR}/tcr_beta_align_out.txt     # TMalign TCR beta chain console output

# Run TMalign on peptide and template peptide structures
/usr/bin/TMalign $pep_path $tpl_pep_path -m $pep_align > $pep_align_out
# Run TMalign on TCR beta chain and template TCR beta chain structures
/usr/bin/TMalign $tcr_beta_path $tpl_tcr_beta_path -m $tcr_align > $tcr_align_out

# Apply TMAlign rotation and translation transformations on pMHC and TCR
# and merge both structures into one Pose
tcr_rechained=${OUT_DIR}/tcr_rechained.pdb
PDB_STEM=$(echo "$OUT_DIR" | rev | cut -d'/' -f -1 | rev)
tcr_pmhc_aligned=${OUT_DIR}/$PDB_STEM.pdb

echo "Aligning complexes..."
tcr_pmhc_align --pmhc_pdb $PMHC --pmhc_tmalign $pep_align --pmhc_tmalign_out $pep_align_out \
--tcr_pdb $tcr_rechained --tcr_tmalign $tcr_align --tcr_tmalign_out $tcr_align_out \
--out_pdb $tcr_pmhc_aligned

# If TM-score from peptide alignment is too small, pdb is not created so exit
if ! (test -f "$tcr_pmhc_aligned"); then
    echo "TM score between $tcr_beta_path and $tpl_tcr_beta_path is too small. Aborting..."
    exit 0
fi

# Run Rosetta docking algorithm on the whole pMHC-TCR complex
# -dock_pert 5 12 means perturbation 5A away from the surface, rotate 12 degrees
echo "Redocking TCR-pMHC complex with NUM_DOCKING=$NUM_DOCKING..."
mkdir -p $OUT_DIR/docked_structures
/Rosetta/main/source/bin/docking_prepack_protocol.default.linuxgccrelease -in:file:s $tcr_pmhc_aligned
/Rosetta/main/source/bin/docking_protocol.linuxgccrelease -in:file:s $tcr_pmhc_aligned -out:path:all $OUT_DIR/docked_structures -overwrite -partners AC_DE -docking_local_refine -dock_pert 5 12 -nstruct $NUM_DOCKING -ex1 -ex2aro

# Interface Analysis
echo "Analyzing complex interface..."
for docked_pdb in $OUT_DIR/docked_structures/*.pdb; do
    /Rosetta/main/source/bin/InterfaceAnalyzer.default.linuxgccrelease -out:level 10 -in:file:s $docked_pdb -interface AC_DE -pack_separated -pack_input -out:file:score_only $score_path
    gzip $docked_pdb
    mv ${docked_pdb}.gz ${OUT_DIR}/
done
rm -rf $OUT_DIR/docked_structures

# Clean all intermediate pdbs
echo "Cleaning all intermediate outputs..."
rm $pep_path $tpl_pep_path $pep_align $pep_align_out $tcr_beta_path $tpl_tcr_beta_path $tcr_align $tcr_beta_align_out $tcr_align_out $tcr_rechained $tcr_pmhc_aligned

# normal exit
exit 0
