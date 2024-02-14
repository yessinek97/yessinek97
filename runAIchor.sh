
# The data should be stored in the input folder in the UI directory
folder_path= #TODO
test_path="/mnt/dataset/$folder_path/test.csv"
train_path="/mnt/dataset/$folder_path/train.csv"
default_configuration= #TODO
experiment_name= #TODO

python -m ig.main train  -train $train_path -test $test_path  -n $experiment_name -c $default_configuration

# Change the folder to download based on tha name chosen above
out="${AICHOR_OUTPUT_PATH}/results/"
folder_to_download="/home/app/biondeep-ig/models"
aws s3 sync $folder_to_download $out --endpoint-url $S3_ENDPOINT
