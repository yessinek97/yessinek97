#############################
##Pull and process raw data##
#############################
# path_row_data_gcp="gs://"
# gsutil cp -r $path_row_data_gcp ./data/debug
# public=./data/debug/Cleanpublic.csv
# optima=./data/debug/optima_20210831_NetMHCpan.tsv
# processing -t $public   -mdp $optima -mdn optima  -c processing_configuration.yml
# optima_pd=./data/debug/optimaPD_20210825_NetMHCpan.tsv
# sahin=./data/debug/sahin2017_20211124_NetMHCpan.tsv
# wells=./data/debug/wellsAdapted_20230207_NetMHCpan.tsv
# processing -odp $optima_pd -odn optima_pd -odp $sahin -odn sahin -odp $wells -odn wells -o proc_data_IG_24_03_2023_CleanPublicNetmhcpan


#############################
##Pull processed data ##
#############################
# proc_gcp_data_path=gs://biondeep-data/IG/data/IG_24_03_2024Netmhcpan/Deafult/proc_data_IG_24_03_2023_CleanPublicNetmhcpan
# destination_path=./data/
# gsutil cp -r $proc_gcp_data_path $destination_path

#############################
##     train command       ##
#############################

dir_path=data/proc_data_IG_24_03_2023_CleanPublicNetmhcpan
dc=debug_default_configuration.yml
train_path=${dir_path}/train.csv
test_path=${dir_path}/optima.csv
optimapd=${dir_path}/optima_pd.csv
wells=${dir_path}/wells.csv
sahin=${dir_path}/sahin.csv

exp_name=debug_Train_command
# train -train $train_path -test $test_path -c $dc -n $exp_name
# inference -d $optimapd -n $exp_name -id id
# inference -d $sahin -n $exp_name -id id
# exp-inference -d $optimapd -n $exp_name
# exp-inference -d $sahin -n $exp_name
# compute-metrics -d $sahin -d $optimapd  -n $exp_name

#############################
##   Multi-train command   ##
#############################

exp_name=debug_Multi_Train_command
c=debug_multi_train_configuration.yml
multi-train -train $train_path -test $test_path -n $exp_name -dc $dc -c $c
multi-inference -d $test_path -d $optimapd -d $wells -d $sahin -mn $exp_name -e -l cd8_any
multi-exp-inference -d $test_path -d $optimapd -d $wells -d $sahin -mn $exp_name

#############################
##Pull quickstart_experiment##
#############################
# quickstart_gcp_data_path=gs://biondeep-data/IG/experiments/quickstart_experiment
# destination_path=./data/
# gsutil cp -r $quickstart_gcp_data_path $destination_path

#############################
##     tune command       ##
#############################
# dir_path=data/proc_data_IG_24_03_2023_CleanPublicNetmhcpan
# tune_config=debug_tune_configuration.yml
# train_path=${dir_path}/train.csv
# test_path=${dir_path}/optima.csv
# exp_name=quickstart_experiment
# tune -train $train_path -test $test_path -c $tune_config -n $exp_name
