#! /bin/bash


# arguments
while getopts :i:o:n: param
do
    case "${param}" in
        i) INPUT_CSV=${OPTARG};;
        o) OUTPUT_DIR=${OPTARG};;
        n) NUM_DOCKING=${OPTARG};;
        *) echo "Unknown flag." 1>&2 && exit 1
    esac
done

if [[ -z "$INPUT_CSV" ]]; then
    echo "Must define path of input csv." 1>&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Must define the output directory." 1>&2
    exit 1
fi
if [[ -z "$NUM_DOCKING" ]]; then
    echo "Must define the number of docked structures generated." 1>&2
    exit 1
fi

mkdir -p $OUTPUT_DIR

set -e

rm_cmds() {
  rm cmds
}

trap "rm_cmds" ERR

# Add empty line at the end of the CSV file if needed
sed -i -e '$a\' $INPUT_CSV

chmod +x ./biondeep_ig/data_gen/pipeline/PosGeneration/gen_tcr_pmhc_poses.sh
# Parse CSV file
# CSV file columns must respect the following order:
# pmhc_filename,tcr_filename,template_filename,optional1,optional2,etc
{
    read
    while IFS="," read -r pmhc_filename tcr_filename template_filename rec_remaining
    do
        echo "./biondeep_ig/data_gen/pipeline/PosGeneration/gen_tcr_pmhc_poses.sh -p $pmhc_filename -r $tcr_filename -t $template_filename -n $NUM_DOCKING -o $OUTPUT_DIR || true" >> cmds
    done
} < $INPUT_CSV

parallel --progress < cmds

# Delete cmds
rm cmds

# Normal exit
exit 0
