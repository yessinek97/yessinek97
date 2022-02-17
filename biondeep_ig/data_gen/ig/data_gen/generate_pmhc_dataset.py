"""Script to generate dataset for pMHC structures."""
from glob import glob
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pandas as pd


def find_wt_crop(wt: str, mut: str) -> str:
    """Crop WT peptide to match tested mutant peptide.

    Args:
        wt: WT peptide sequence.
        mut: Mutant peptide sequence.

    Returns:
        cropped WT peptide that matched mutant.
    """
    best, best_index = 0, 0
    for i in range(0, len(wt) - len(mut) + 1):
        similarity = (np.array(list(wt[i : i + len(mut)])) == np.array(list(mut))).sum() / len(mut)
        if similarity > best:
            best = similarity
            best_index = i
    return wt[best_index : best_index + len(mut)]


def merge_datasets(
    paths: List[str],
    mut_column: str = "tested_peptide_biondeep_MHCI",
    wt_column: str = "wt_peptide",
    allele_column: str = "tested_allele_biondeep_MHCI",
) -> pd.DataFrame:
    """Clean and merge datasets into one DataFrame.

    Args:
        paths: list of datasets paths.
        mut_column: mutant peptide column in datasets.
        wt_column: WT peptide column in datasets.
        allele_column: allele column in datasets.

    Returns:
        dataframe containing the combinations of allele-wt-mut.
    """
    dfs = [
        pd.read_csv(path).dropna(subset=[mut_column, wt_column, allele_column]) for path in paths
    ]

    for df in dfs:
        df["cropped_wt_peptide"] = df.apply(
            lambda row: find_wt_crop(row[wt_column], row[mut_column]),
            axis=1,
        )
        df["allele-wt-mut"] = df.apply(
            lambda row: row[allele_column]
            + "-"
            + row[mut_column]
            + "-"
            + row["cropped_wt_peptide"],
            axis=1,
        )
        df["mut-allele-peptide"] = df.apply(
            lambda row: row[allele_column] + "-" + row[mut_column],
            axis=1,
        )
        df["wt-allele-peptide"] = df.apply(
            lambda row: row[allele_column] + "-" + row["cropped_wt_peptide"],
            axis=1,
        )

    return pd.concat(dfs)


def get_allele_peptide_tuples(
    df: pd.DataFrame,
    mut_column: str = "tested_peptide_biondeep_MHCI",
    wt_column: str = "cropped_wt_peptide",
    allele_column: str = "tested_allele_biondeep_MHCI",
) -> pd.DataFrame:
    """Get all DataFrame of all (allele, peptide) tuples.

    Args:
        df: input dataframe
        mut_column: mutant peptide column in datasets.
        wt_column: WT peptide column in datasets.
        allele_column: allele column in datasets.

    Returns:
        dataframe of all (allele, peptide) tuples.
    """
    wt_df = (
        df[[allele_column, wt_column]]
        .drop_duplicates()
        .rename(columns={allele_column: "allele", wt_column: "peptide"})
    )
    mut_df = (
        df[[allele_column, mut_column]]
        .drop_duplicates()
        .rename(columns={allele_column: "allele", mut_column: "peptide"})
    )

    return pd.concat([wt_df, mut_df]).drop_duplicates()


def get_allele_filename_mapping(pmhc_dir: str, pipeline_prefix_dir: str) -> Dict[str, str]:
    """Get allele to PDB mapping.

    Requires that the pMHC filenames starts with the following naming conventions:
    <ALLELE>.<PEPTIDE>*.pdb

    Args:
        pmhc_dir: dir containing pMHC structures.
        pipeline_prefix_dir: prefix to add (where the generated pMHC will be stored).

    Returns:
        mapping between allele and corresponding structures.
    """
    allele_filename: Dict[str, Any] = {}
    for filepath in glob(str(Path(pmhc_dir) / "*.pdb")):
        try:
            allele = Path(filepath).name.split(".")[0]
        except Exception as e:
            raise ValueError(
                f"{filepath} doesn't follow the naming convention: <ALLELE>.<PEPTIDE>*.pdb"
            ) from e
        allele_filename[allele] = allele_filename.get(allele, list(map(str, []))) + [
            str(Path(pipeline_prefix_dir) / Path(filepath).name)
        ]

    return allele_filename


def duplicate_allele_peptide(df: pd.DataFrame, num_flags: int) -> pd.DataFrame:
    """Replicate dataframe to account for the number of flags.

    Args:
        df: dataframe of allele-peptide-filename.
        num_flags: number of flags.

    Returns:
        dataframe with duplicated entries with unique flags
    """
    df = pd.concat([df] * num_flags).sort_values(by="peptide")
    counters = {
        k: 0 for k in df.apply(lambda x: x["allele"] + "-" + x["peptide"], axis=1).to_list()
    }
    flags = []
    for _, row in df.iterrows():
        tmp = row["allele"] + "-" + row["peptide"]
        flags.append(counters[tmp])
        counters[tmp] += 1
    df["FLAG"] = flags
    return df


def add_allele_filename(
    df: pd.DataFrame,
    pmhc_dir: str,
    num_flags: int = 4,
    pipeline_prefix_dir: str = "/mnt/data/ig-dataset/artefacts/pmhc",
) -> pd.DataFrame:
    """Compute allele-filename and replicate dataframe.

    Args:
        df: dataframe with allele and peptide columns.
        pmhc_dir: dir containing pMHC structures.
        num_flags: number of flags.
        pipeline_prefix_dir: prefix to add (where the generated pMHC will be stored).

    Returns:
        dataframe.
    """
    allele_filename = get_allele_filename_mapping(pmhc_dir, pipeline_prefix_dir)

    new_df_data = []
    for _, row in df.iterrows():
        allele = row["allele"]
        if allele not in allele_filename:
            continue
        filenames = allele_filename[allele]
        for filename in filenames:
            new_df_data.append(
                {
                    "filename": filename,
                    "allele": allele,
                    "peptide": row["peptide"],
                }
            )

    new_df = pd.DataFrame(new_df_data)

    return duplicate_allele_peptide(new_df, num_flags)


def generate_pmhc_dataset(
    paths: List[str],
    pmhc_dir: str,
    num_flags: int = 4,
    mut_column: str = "tested_peptide_biondeep_MHCI",
    wt_column: str = "wt_peptide",
    allele_column: str = "tested_allele_biondeep_MHCI",
    pipeline_prefix_dir: str = "/mnt/data/ig-dataset/artefacts/pmhc",
) -> pd.DataFrame:
    """Generate pMHC dataset.

    Args:
        paths: list of datasets paths.
        pmhc_dir: dir containing pMHC structures.
        num_flags: number of flags.
        mut_column: mutant peptide column in datasets.
        wt_column: WT peptide column in datasets.
        allele_column: allele column in datasets.
        pipeline_prefix_dir: prefix to add (where the generated pMHC will stored).

    Returns:
        dataframe containing allele, peptide, template path and flag columns
        (input to pipeline)
    """
    df = merge_datasets(
        paths=paths,
        mut_column=mut_column,
        wt_column=wt_column,
        allele_column=allele_column,
    )
    df = get_allele_peptide_tuples(
        df=df,
        mut_column=mut_column,
        wt_column=wt_column,
        allele_column=allele_column,
    )
    df = add_allele_filename(
        df=df,
        pmhc_dir=pmhc_dir,
        num_flags=num_flags,
        pipeline_prefix_dir=pipeline_prefix_dir,
    )

    return df
