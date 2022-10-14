test_path=data/IG_19_08_2022_transformer/IG_19_08_2022_transformer.optima.Biondeep.clean.csv
train_path=data/IG_19_08_2022_transformer/IG_19_08_2022_transformer.public.Biondeep.clean.csv

train  -train $train_path -test $test_path  -n Trf_public_v2.bnt_19_08_2022 -c transformer.bnt.yml
train  -train $train_path -test $test_path  -n Trf_public_v2.bnt.pmhc_19_08_2022 -c transformer.bnt.pmhc.yml
train  -train $train_path -test $test_path  -n Trf_public_v2.bnt.biondeep_19_08_2022 -c transformer.bnt.biondeep.yml
train  -train $train_path -test $test_path  -n Trf_public_v2.bnt.biondeep.pmhc_19_08_2022 -c transformer.bnt.biondeep.pmhc.yml

test_path=data/IG_19_08_2022_Netmhcipan_V3/IG_19_08_2022_Netmhcipan_V3.optima.clean.csv
train_path=data/IG_19_08_2022_Netmhcipan_V3/IG_19_08_2022_Netmhcipan_V3.public.clean.csv

train  -train $train_path -test $test_path  -n Net_public_v3.bnt_19_08_2022 -c netmhcipan.bnt_v3.yml
train  -train $train_path -test $test_path  -n Net_public_v3.bnt.pmhc_19_08_2022 -c netmhcipan.bnt.pmhc_v3.yml
train  -train $train_path -test $test_path  -n Net_public_v3.bnt.netmhcpan_19_08_2022 -c netmhcipan.bnt.netmhcpan_v3.yml
train  -train $train_path -test $test_path  -n Net_public_v3.bnt.netmhcpan.pmhc_19_08_2022 -c netmhcipan.bnt.netmhcpan.pmhc_v3.yml
train  -train $train_path -test $test_path  -n Net_public_v3.netmhcpan.pmhc_19_08_2022 -c netmhcipan.netmhcpan.pmhc_v3.yml
