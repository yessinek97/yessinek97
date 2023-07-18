data_path=data/cimt/
public_data_path=${data_path}IG_16_11_2022_Netmhcpan_public_go_embeddings_rna_representation.csv
main_configuration_file=CIMT_IG_16_11_2022.yml
exp_name=CIMT_NetMHCpan_IG_16_11_2022_Validation
cimt  -d $public_data_path -c $main_configuration_file -n $exp_name
# cimt-inference -d $public_data_path -n $exp_name -e
