#!/bin/bash

# The folder path should be stored in the input folder in the UI directory
data_folder="/mnt/dataset/"
folder_path= #TODO
test_path="${data_folder}${folder_path}/optima.csv" #test set
train_path="${data_folder}${folder_path}/train.csv" #train set

# Download dataset from S3
aws s3 sync ${AICHOR_INPUT_PATH}${folder_path} ${data_folder}${folder_path} --endpoint-url $S3_ENDPOINT

# List downloaded files for verification
ls ${data_folder}${folder_path}

#  Train the model
python -m ig.main multi-train-distributed  --train_data_path  $train_path --test_data_path  $test_path  -c multi_train_configuration.yml -dc default_configuration.yml  -n  multi_train_exp


#Change the folder to download based on tha name chosen above
out="${AICHOR_OUTPUT_PATH}results/"
folder_to_download="/home/app/biondeep-ig/models"
aws s3 sync $folder_to_download $out --endpoint-url $S3_ENDPOINT
