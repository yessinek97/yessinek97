
# The data should be stored in the input folder in the UI directory
test_path="/mnt/dataset/quick_start/test.csv"
train_path="/mnt/dataset/quick_start/train.csv"

python -m ig.main train  -train $train_path -test $test_path  -n test_quick_start_ichor -c quickstart_train.yml

# Change the folder to download based on tha name chosen above
out="${AICHOR_OUTPUT_PATH}/results/"
folder_to_download="/home/models/"
aws s3 sync $folder_to_download $out --endpoint-url $S3_ENDPOINT
