# The data should be stored in the input folder in the UI directory
data_folder="/mnt/dataset/"
test_path="/mnt/dataset/IG16112022Biondeep/optima.clean.go.csv"
train_path="/mnt/dataset/IG16112022Biondeep/public.clean.go.csv"


aws s3 sync ${AICHOR_INPUT_PATH}IG16112022Biondeep/ ${data_folder}IG16112022Biondeep/ --endpoint-url $S3_ENDPOINT
ls /mnt/dataset/IG16112022Biondeep/

python -m ig.main multi-train-distributed  -train $train_path  -test $test_path -c multi_train_configuration.yml -dc default_configuration.yml  -n  multi_train_exp

#Change the folder to download based on tha name chosen above
out="${AICHOR_OUTPUT_PATH}results/"
folder_to_download="/home/models/"
aws s3 sync $folder_to_download $out --endpoint-url $S3_ENDPOINT
