# TCR Structure Generation

_Note_: The following command lines must be executed within the Docker container provided.

## Full alpha/beta chain

We use `/Rosetta/main/source/bin/tcrmodel.default.linuxgccrelease` from
[Rosetta](https://www.rosettacommons.org/docs/latest/application_documentation/structure_prediction/TCRmodel).
Example usage is

```bash
/Rosetta/main/source/bin/tcrmodel.default.linuxgccrelease -alpha $alpha -beta $beta -out:suffix ${suffix} -out:level 400 -out:pdb_gz -out:path:all ${output_dir}
```

where, user should provide full alpha and beta amino acid sequences.

## CDR3a/b chain

Optionally, we can use the provided bash script with only cdr3a and cdr3b sequences,

```bash
chmod +x ./biondeep_ig/data_gen/ig/data_gen/tcr/tcrmodel.sh
./biondeep_ig/data_gen/ig/data_gen/tcr/tcrmodel.sh -a $cdr3a -b $cdr3b -o ${output_dir}
```

where we consider the following full alpha/beta chain,

> **Alpha Chain**
>
> AQTVTQSQPEMSVQEAETVTLSCTYDTSENDYILFWYKQPPSRQMILVIRQEAYKQQNATENRFSVNFQKAAKSFSLKISDSQLGDAAMYF**CAYGEDDKIIF**GKGTRLHILPNIQNPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKCVLDMRSMDFKSNSAVAWSNKSDFACANAFNN
>
> **Beta Chain**
>
> AEADIYQTPRYLVIGTGKKITLECSQTMGHDKMYWYQQDPGMELHLIHYSYGVNSTEKGDLSSESTVSRIRTEHFPLTLESARPSHTSQYL**CASRRGSAELYF**GPGTRLTVTEDLKNVFPPEVAVFEPSEAEISHTQKATLVCLATGFYPDHVELSWWVNGKEVHSGVCTDPQPLKEQPALNDSRYALSSRLRVSATFWQDPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRAA

and the `CAYGEDDKIIF` in alpha chain is changed to the given CDR3a and the `CASRRGSAELYF` in beta
chain is changed to the given CDR3b.
