"""Script to merge tcr-pMHC structure features into a given dataset."""
from pathlib import Path

import click
import numpy as np

from biondeep_ig.src.logger import get_logger
from biondeep_ig.src.logger import init_logger
from biondeep_ig.src.utils import load_yml
from biondeep_ig.src.utils import read_data

log = get_logger("tcr-pMHC data merger")

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CFG_FILE = str(SCRIPT_DIR / "config_merge_tcrpmhc.yml")


@click.command()
@click.option(
    "--input-dataset",
    "-i",
    type=str,
    required=True,
    help="Path to the dataset used in training.",
)
@click.option(
    "--tcr-pmhc-dataset",
    "-t",
    type=str,
    required=True,
    help="Path to the tcr-pMHC dataset.",
)
@click.option(
    "--output-dir",
    "-o",
    type=str,
    required=True,
    help="Path to the output directory.",
)
@click.option(
    "--config_file",
    "-c",
    type=str,
    default=DEFAULT_CFG_FILE,
    help="Path to the configuration file.",
)
def command(input_dataset, tcr_pmhc_dataset, output_dir, config_file):
    """Click command that calls main().

    Args:
        input_dataset (str): Path to the input dataset.
        tcr_pmhc_dataset (str): Path to the tcr-pMHC dataset.
        output_dir (str): Path to the output directory.
        config_file (str): Path to the config file.
    """
    main(input_dataset, tcr_pmhc_dataset, output_dir, config_file)


def main(input_dataset, tcr_pmhc_dataset, output_dir, config_file=DEFAULT_CFG_FILE):
    """Merge the tcr-pMHC structure features into a given dataset.

    Args:
        input_dataset (str): Path to the input dataset.
        tcr_pmhc_dataset (str): Path to the tcr-pMHC dataset.
        output_dir (str): Path to the output directory.
        config_file (str): Path to the config file.
    """
    log_file = str(Path(output_dir) / "InfoTCRDataMerger.log")
    init_logger(folder_name=None, file_name=log_file)

    config = load_yml(config_file)

    data_tcr = read_data(tcr_pmhc_dataset)
    df = read_data(input_dataset)

    data_tcr = clean_tcr_pmhc_data(data_tcr)
    # Preprocessing datasets: removing previously structure features
    data_tcr.columns = data_tcr.columns.str.lower()
    tcr_columns = list(data_tcr.columns)
    df = filter_cols(df, tcr_columns)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_dataset_prefix = output_dir / Path(input_dataset).stem
    generate_data(
        tcr_ds=data_tcr,
        binding_energy_column=config["binding_energy_column"],
        init_ds=df,
        output_dataset_prefix=output_dataset_prefix,
        right_on=config["right_on"],
        left_on=config["left_on"],
        select_on=config["select_on"],
    )

    log.info(
        "The following features have been added, "
        "make sure to include the relevant ones in your features list before training/testing:"
    )
    for feat in tcr_columns:
        log.info(feat)


def clean_tcr_pmhc_data(data_tcr):
    """Clean tcr-pMHC data.

    Args:
        data_tcr (pd.DataFrame): tcr-pMHC dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    data_tcr.loc[data_tcr.dG_separated > 0, "dG_separated"] = 0
    data_tcr.loc[data_tcr["dG_separated/dSASAx100"] > 0, "dG_separated/dSASAx100"] = 0
    data_tcr = data_tcr[~data_tcr.dG_separated.isna()]
    tcr_columns_nan = data_tcr.isna().mean()
    tcr_columns_nan = tcr_columns_nan[tcr_columns_nan > 0.7].index.tolist() + [
        "dG_cross/dSASAx100",
        "dG_cross",
        "packstat",
        "total_score",
        "description",
        "allele-peptide",
    ]
    data_tcr.drop(tcr_columns_nan, inplace=True, axis=1)
    return data_tcr


def filter_cols(df, drop_list):
    """Remove columns from dataframe if they exist.

    Args:
        df (pd.DataFrame): A DataFrame that may already contain some columns to be dropped.
        drop_list (list(str)): List of columns to drop.

    Returns:
        pd.DataFrame: A DataFrame from which the desired columns have been dropped.
    """
    my_filter = df.filter(drop_list)
    return df.drop(my_filter, axis=1)


def aggregate_on_quantile(tcr_pmhc_data, quantile_ratio, right_on, binding_energy_column):
    """Aggregate the tcr-pMHC features by averaging in the same binding energy quantile.

    For each allele or peptide-allele pair, we have multiple tcr- pMHC-TCR structures.
    Aggregation is necessary to generate features per peptide-allele pair.
    Given N structures:
    The structures having dG_separated values within a given quantile are selected.
    The features corresponding to the selected structures are averaged.
    The averaged features are used as the aggregated features.

    Args:
        tcr_pmhc_data (pd.DataFrame): tcr-pMHC dataset.
        quantile_ratio (float): quantile value.
        right_on (list(str)): columns name from the tcr-pMHC data are aggregated (e.g. ["allele"]
        or ["allele", "peptide"])
        binding_energy_column (str): column name of the binding energy (e.g. "dG_separated").

    Returns:
        pd.DataFrame: DataFrame containing the aggregated features.
    """

    def quantile(x):
        return np.quantile(x, quantile_ratio)

    aggregated_data = tcr_pmhc_data.merge(
        tcr_pmhc_data.groupby(right_on)[binding_energy_column]
        .apply(quantile)
        .rename("dg_threshold")
        .reset_index(),
        on=right_on,
        how="left",
    )
    aggregated_data = aggregated_data[
        aggregated_data[binding_energy_column] <= aggregated_data.dg_threshold
    ]
    aggregated_data = aggregated_data.drop(["dg_threshold"], axis=1)
    aggregated_data = aggregated_data.groupby(right_on).mean().reset_index()
    return aggregated_data


def merge_datasets(dg_data, right_on, left_on, target_dataset, binding_energy_column):
    """Merge the tcr-pMHC features to a given dataset.

    Args:
        dg_data (pd.DataFrame): tcr-pMHC features aggregated along binding energy.
        right_on (str): column name from the tcr-pMHC data on which the merge is performed.
        left_on (str): column name from the target dataset on which the merge is performed.
        target_dataset (pd.DataFrame): target dataset.
        binding_energy_column (str): column name of the binding energy (e.g. "dG_separated").

    Returns:
        pd.DataFrame: Dataset containing the tcr-pMHC structure features.
    """
    result_dg = target_dataset.merge(dg_data, how="left", right_on=right_on, left_on=left_on)
    for col in right_on:
        # Remove column from right df that was used to merge
        if col in result_dg.columns:
            result_dg.drop(col, axis=1, inplace=True)
        # If target_dataset contained a column with the same name as "right_on" keyword,
        # remove duplicated columns created by pandas
        if f"{col}_x" in result_dg.columns:
            result_dg.drop(f"{col}_x", axis=1, inplace=True)
        if f"{col}_y" in result_dg.columns:
            result_dg.drop(f"{col}_y", axis=1, inplace=True)

    log.info(f"Length of original vs processed datasets: {len(target_dataset)}  {len(result_dg)}")

    nb_nans = result_dg[binding_energy_column].isna().sum()
    if nb_nans > 0:
        log.info(f"There are {nb_nans} nan in the merged data")

    return result_dg


def generate_data(
    tcr_ds, binding_energy_column, init_ds, output_dataset_prefix, right_on, left_on, select_on
):
    """Merge raw dataset with tcr-pMHC data and write to csv files.

    Several aggregation strategy are applied:
    - minimum of binding energy
    - averaging over binding energy quantile

    Args:
        tcr_ds (pd.DataFrame): tcr-pMHC dataset features.
        binding_energy_column (str): Column name corresponding to the binding energy score.
        init_ds (pd.DataFrame): Raw dataset containing all features.
        output_dataset_prefix (str): Prefix path used to save the output datasets.
        right_on (list(str)): List of columns name from the tcr-pMHC data
                              on which the merge is performed.
        left_on (list(str)): List of columns name the target dataset
                             on which the merge is performed.
        select_on (str): Statistics applied to binding energy use to select and aggregate features.
    """
    log.info(f"Binding energy: {binding_energy_column}")
    log.info(f"Merging columns: left={' '.join(left_on)}, right={' '.join(right_on)}")
    if select_on == "min":
        log.info(f"Aggregation: minimum of {binding_energy_column}")
        dg_data = tcr_ds.loc[tcr_ds.groupby(right_on)[binding_energy_column].idxmin()].reset_index(
            drop=True
        )
        result_dg = merge_datasets(
            dg_data,
            right_on,
            left_on,
            target_dataset=init_ds,
            binding_energy_column=binding_energy_column,
        )
        result_dg.to_csv(
            f"{output_dataset_prefix}"
            f"_{select_on}_{binding_energy_column}_{'_'.join(right_on)}.csv",
            index=False,
        )
    elif select_on == "mean":
        log.info(f"Aggregation: Mean of {binding_energy_column}")
        dg_data = aggregate_on_quantile(tcr_ds, 1, right_on, binding_energy_column)
        result_dg = merge_datasets(
            dg_data,
            right_on,
            left_on,
            target_dataset=init_ds,
            binding_energy_column=binding_energy_column,
        )
        result_dg.to_csv(
            f"{output_dataset_prefix}_mean_{binding_energy_column}_{'_'.join(right_on)}.csv",
            index=False,
        )
    elif isinstance(select_on, float):
        quantile_ratio = select_on
        quantile = int(select_on * 100)
        log.info(f"Aggregation: {quantile}th quantile of {binding_energy_column}")
        dg_data = aggregate_on_quantile(tcr_ds, quantile_ratio, right_on, binding_energy_column)
        result_dg = merge_datasets(
            dg_data,
            right_on,
            left_on,
            target_dataset=init_ds,
            binding_energy_column=binding_energy_column,
        )
        result_dg.to_csv(
            f"{output_dataset_prefix}"
            f"_{quantile}quantile_{binding_energy_column}_{'_'.join(right_on)}.csv",
            index=False,
        )
    else:
        raise ValueError(f"Unknown selection method: {select_on}")
