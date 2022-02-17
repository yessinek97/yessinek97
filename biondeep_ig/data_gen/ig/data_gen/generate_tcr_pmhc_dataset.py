"""Script to generate dataset for TCR-pMHC structures."""
import copy
import random
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from shutil import rmtree
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

RECEPTOR_URL = "http://www.iedb.org/downloader.php?file_name=doc/receptor_full_v3.zip"
TCELL_URL = "http://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip"


def download_and_unzip_data(url: str, out_path: str):
    """Download and unzip data from IEDB.

    Args:
        url: url to csv data.
        out_path: where to extract the downloaded zip.
    """
    try:
        urlretrieve(url, out_path)
        with zipfile.ZipFile(out_path, "r") as zip_ref:
            zip_ref.extractall(str(Path(out_path).parent))
    except URLError as e:
        raise RuntimeError(f"Unable to download {url} - reason: {e.reason}") from e


def download_iedb_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download data from IEDB.

    Returns:
        dataframes for receptor and tcell data.
    """
    tmp_dir = tempfile.mkdtemp()

    receptor_path = Path(tmp_dir) / RECEPTOR_URL.rsplit("/", maxsplit=1)[-1]
    tcell_path = Path(tmp_dir) / TCELL_URL.rsplit("/", maxsplit=1)[-1]
    download_and_unzip_data(
        url=RECEPTOR_URL,
        out_path=str(receptor_path),
    )
    download_and_unzip_data(
        url=TCELL_URL,
        out_path=str(tcell_path),
    )

    receptor_df = pd.read_csv(str(receptor_path.with_suffix(".csv")))
    tcell_df = pd.read_csv(str(tcell_path.with_suffix(".csv")))

    rmtree(tmp_dir)

    return receptor_df, tcell_df


def get_assay_tcr_mapping(receptor_df: pd.DataFrame) -> Dict[str, str]:
    """Get mapping between Assay and TCR in IEDB.

    Args:
        receptor_df: pd.DataFrame.

    Returns:
        mapping between assay and TCR.
    """
    receptor_df.dropna(subset=["Chain 1 CDR3 Curated", "Chain 2 CDR3 Curated"], inplace=True)
    receptor_df["CDR3a-CDR3b"] = receptor_df.apply(
        lambda row: str(row["Chain 1 CDR3 Curated"]) + "-" + str(row["Chain 2 CDR3 Curated"]),
        axis=1,
    )

    assay_tcr = {}
    for _, row in receptor_df.iterrows():
        assays = [str(int(i)) for i in str(row["Assay IDs"]).split(",")]
        tcr = row["CDR3a-CDR3b"]
        for assay in assays:
            assay_tcr[assay] = tcr

    return assay_tcr


def get_tcr_template_mapping(tm_df: pd.DataFrame) -> pd.DataFrame:
    """Get mapping between TCR and TCR-pMHC based on TM score.

    The dataframe must have the following format:
                                     tcr          template       tm
    0	tcr-CAARLYGGSQGNLIF-CSARDWGYEQYF	5BS0.rechained	0.86955
    1	tcr-CAARLYGGSQGNLIF-CSARDWGYEQYF	3D3V.rechained	0.94165
    ...
    15174	tcr-CAASGTLTTSGTYKYIF-CASSQEGTAYEQYF	4PRP.rechained	0.86059
    15175	tcr-CAASGTLTTSGTYKYIF-CASSQEGTAYEQYF	3W0W.rechained	0.92716

    Args:
        tm_df: dataframe containing alignment scores.

    Returns:
        dataframe with said mapping.
    """
    agg_tm_df = tm_df.groupby("tcr").max("tm").reset_index()

    def find_template(x):
        tcr, tm = x["tcr"], x["tm"]
        return tm_df[(tm_df["tcr"] == tcr) & (tm_df["tm"] == tm)]["template"].to_list()[0]

    agg_tm_df["template"] = agg_tm_df.apply(find_template, axis=1)

    return agg_tm_df


def prepare_tcell_df(tcell_df: pd.DataFrame, assay_tcr: Dict[str, str]) -> pd.DataFrame:
    """Prepare tcell DataFrame.

    Args:
        tcell_df: dataframe with IEDB T-cell data.
        assay_tcr: mapping between assay and TCR.

    Returns:
        prepared tcell dataframe.
    """
    # Filter on MHC type
    def select_allele(x):
        try:
            return x.split("-")[1].replace("*", "").replace(":", "")
        except IndexError:
            return np.nan

    # TODO: adapt code for MHC-II structures
    tcell_mhc1_df = tcell_df[tcell_df["MHC.1"] == "I"]
    tcell_mhc1_df["allele"] = tcell_mhc1_df["MHC"].apply(select_allele)
    tcell_mhc1_df.dropna(subset=["allele"], inplace=True)
    # Retrieve assay
    tcell_mhc1_df["assay"] = tcell_mhc1_df["Reference"].apply(lambda x: x.split("/")[-1])
    # Get TCR for every assay
    tcell_mhc1_df["tcr"] = tcell_mhc1_df["assay"].apply(
        lambda x: assay_tcr[str(x)] if str(x) in assay_tcr else np.nan
    )
    tcell_mhc1_df.dropna(subset=["tcr"], inplace=True)

    return tcell_mhc1_df


def get_allele_tcr_mapping(tcell_df: pd.DataFrame, tm_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get mapping between allele and TCR.

    Args:
        tcell_df: dataframe with IEDB T-cell data.
        tm_df: dataframe with alignment scores.

    Returns:
        mapping between allele and TCR.
    """
    tcrs = list(map(lambda tcr: tcr[4:], tm_df["tcr"].unique()))

    allele_tcr: Dict[str, List[str]] = {}
    for _, row in tcell_df.dropna(subset=["tcr"]).iterrows():
        if str(row["tcr"]) in tcrs:
            allele_tcr[row["allele"]] = allele_tcr.get(row["allele"], []) + [row["tcr"]]

    def get_list(v):
        x = sorted(v, key=Counter(v).get, reverse=True)
        return sorted(set(x), key=x.index)

    allele_tcr = {k: get_list(v) for k, v in allele_tcr.items()}
    allele_tcr_filtered = {k: v for k, v in allele_tcr.items() if k is not None}

    return allele_tcr_filtered


def get_most_frequent_tcrs(receptor_df: pd.DataFrame) -> List[str]:
    """Get list of TCRs sorted by frequency.

    Args:
        receptor_df: dataframe with IEDB receptor data.

    Returns:
        list of most frequent TCRs.
    """
    most_frequent_tcrs = []
    for tcr in receptor_df["CDR3a-CDR3b"].value_counts().index:
        most_frequent_tcrs.append(tcr)
    return most_frequent_tcrs


def get_similar_alleles(allele: str, alleles: List[str]) -> List[str]:
    """Get list of same-group/locus alleles for allele.

    e.g
    Args:
        allele: allele id.
        alleles: list of all allele ids.

    Returns:
        list of similar alleles sorted by similarity.
    """
    similar_alleles = [al for al in alleles if al[:3] == allele[:3]]
    similar_alleles.extend(
        [al for al in alleles if al[0] == allele[0] or al not in similar_alleles]
    )
    return similar_alleles


def select_tcrs_for_allele(
    allele: str, allele_tcr: Dict[str, List[str]], most_frequent_tcrs: List[str], size: int = 20
) -> List[Tuple[str, str]]:
    """Select list of TCRs for allele.

    Args:
        allele: allele id.
        allele_tcr: mapping between allele and TCR.
        most_frequent_tcrs: list of most frequent TCRs.
        size: number of TCRs to be tested per allele.

    Returns:
        list of TCRs for allele.
    """
    tcrs = []
    # 1. Check if allele is in IEDB
    if allele in allele_tcr:
        tcrs = [(tcr, "selected from given allele in IEDB") for tcr in allele_tcr[allele]]
    if len(tcrs) >= size:
        return tcrs[:size]
    # 2. Get similar alleles and select corresponding TCRs
    all_alleles = list(
        filter(
            lambda all: all != allele and bool(re.match(r"[A-Z]{1}[0-9]{4}.*", all)),
            list(allele_tcr.keys()),
        )
    )
    similars = get_similar_alleles(
        allele=allele,
        alleles=all_alleles,
    )
    for similar in similars:
        tcrs += [
            (tcr, f"selected from similar alleles {similar} in IEDB") for tcr in allele_tcr[similar]
        ]
        if len(tcrs) >= size:
            return tcrs[:size]
    # 3. Get most frequent TCRs in IEDB
    top_tcrs = copy.deepcopy(most_frequent_tcrs[:size])
    random.shuffle(top_tcrs)
    tcrs += [
        (tcr, "random from most frequent TCRs in IEDB") for tcr in top_tcrs[: size - len(tcrs)]
    ]
    if len(tcrs) < size:
        raise ValueError(f"Not enough TCRs were selected: {len(tcrs)}. Requested {size}.")
    return tcrs


def generate_tcr_pmhc_dataset(
    pmhc_df: pd.DataFrame,
    tm_alignment_path: str,
    artefacts_dir: str = "/mnt/data/ig-dataset/artefacts",
    include_min_pmhcs: bool = False,
) -> pd.DataFrame:
    """Generate TCR-pMHC dataset.

    Args:
        pmhc_df: pMHC data from pMHC dataset generation.
        tm_alignment_path: path to TCR-template alignment csv.
        artefacts_dir: path to artefacts in pipeline.

    Returns:
        dataframe with pMHC filename, TCR filename and template filename columns.
    """
    receptor_df, tcell_df = download_iedb_data()

    # Get assay-TCR mapping from receptor data
    assay_tcr = get_assay_tcr_mapping(receptor_df=receptor_df)

    tcell_df = prepare_tcell_df(tcell_df=tcell_df, assay_tcr=assay_tcr)

    tm_align_df = pd.read_csv(tm_alignment_path, error_bad_lines=False)
    tm_align_df["tm"] = pd.to_numeric(tm_align_df["tm"])

    agg_tm_df = get_tcr_template_mapping(tm_df=tm_align_df)

    allele_tcr = get_allele_tcr_mapping(tcell_df=tcell_df, tm_df=tm_align_df)

    most_frequent_tcrs = [
        tcr
        for tcr in get_most_frequent_tcrs(receptor_df=receptor_df)
        if f"tcr-{tcr}" in tm_align_df["tcr"].unique()
    ]

    alleles = pmhc_df["allele"].unique()
    allele_tcr_data: List[Dict[str, Union[str, List]]] = [{"allele": allele} for allele in alleles]
    for i, allele in enumerate(alleles):
        allele_tcr_data[i].update(  # type: ignore
            {
                "tcrs": list(
                    map(
                        lambda tcr: tcr[0],
                        select_tcrs_for_allele(
                            allele=allele,
                            allele_tcr=allele_tcr,
                            most_frequent_tcrs=most_frequent_tcrs,
                        ),
                    )
                )
            }
        )
    allele_tcr_df = pd.DataFrame(allele_tcr_data)

    tcr_pmhc_data = []
    for _, row in pmhc_df.iterrows():
        init_pdb_name = str(Path(row["filename"]).stem)
        tcrs = allele_tcr_df[allele_tcr_df["allele"] == row["allele"]]["tcrs"].to_list()
        if len(tcrs) == 0:
            tcrs = [most_frequent_tcrs[:20]]
        for tcr in tcrs[0]:
            tcr = f"tcr-{tcr}"
            template = agg_tm_df[agg_tm_df["tcr"] == tcr]["template"].to_list()
            if len(template) == 0:  # choose random TCR/template
                index = np.random.choice(len(agg_tm_df))
                template = [agg_tm_df.iloc[index]["template"]]
                tcr = agg_tm_df.iloc[index]["tcr"]
            if include_min_pmhcs:
                tcr_pmhc_data.extend(
                    [
                        {
                            "filename": str(
                                Path(artefacts_dir).parent
                                / "generated_pmhc"
                                / f"{init_pdb_name}_{row['peptide']}_{row['FLAG']}_min.pdb.gz"
                            ),
                            "tcr": str(Path(artefacts_dir) / "tcr" / f"{tcr}.pdb"),
                            "template": str(
                                Path(artefacts_dir) / "template" / f"{template[0]}.pdb"
                            ),
                        },
                    ]
                )
            tcr_pmhc_data.extend(
                [
                    {
                        "filename": str(
                            Path(artefacts_dir).parent
                            / "generated_pmhc"
                            / f"{init_pdb_name}_{row['peptide']}_{row['FLAG']}_relax.pdb.gz"
                        ),
                        "tcr": str(Path(artefacts_dir) / "tcr" / f"{tcr}.pdb"),
                        "template": str(Path(artefacts_dir) / "template" / f"{template[0]}.pdb"),
                    },
                ]
            )

    df = pd.DataFrame(tcr_pmhc_data)
    df.rename(
        columns={
            "filename": "pmhc_filename",
            "tcr": "tcr_filename",
            "template": "template_filename",
        },
        inplace=True,
    )

    return df
