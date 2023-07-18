"""The goal is to generate the peptide-allele paris from different datasets."""
import pathlib
from logging import Logger
from typing import Any, List, Union

import click
import pandas as pd

from ig.src.logger import get_logger
from ig.src.utils import read_data

log: Logger = get_logger("Generate pepetide allele pairs")


@click.command()
@click.option(
    "--data_paths",
    "-d",
    type=str,
    required=True,
    multiple=True,
    help="data path.",
)
@click.option(
    "--peptide",
    "-p",
    type=str,
    required=True,
    help="peptide column name.",
)
@click.option(
    "--allele",
    "-a",
    type=str,
    required=True,
    help="allele column name.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    required=True,
    help="output file path and name.",
)
@click.pass_context
def generate_pairs(
    ctx: Union[click.core.Context, Any],  # pylint: disable=unused-argument
    data_paths: str,
    peptide: str,
    allele: str,
    output: str,
) -> None:
    """Generate the peptide-allele paris for one or multiple datasets.

    Args:
        data_paths: path to single or multiple datasets tsv or csv.
        peptide: peptide column name, should be the same for all datasets.
        allele: allele column name, should be the same for all datasets.
        output: where the output file should be generated.

    Returns:
        generate a file.

    """
    log.info("Starting generation!")
    data_final: List[pd.DataFrame] = []
    for data_path in data_paths:
        data = read_data(data_path)
        # lower columns names
        data.columns = data.columns.str.lower()
        data["name_data"] = pathlib.Path(data_path).name
        pair_data = data[["id", "name_data", allele, peptide]]
        data_final.append(pair_data)
    data_final_csv = pd.concat(data_final, axis=0)
    log.info("Length of dataframe BEFORE removing duplicates: %s", len(data_final_csv))
    data_final_csv = data_final_csv.drop_duplicates(subset=[allele, peptide])
    # remove subest which do not have any specific values for peptide or allele.
    data_final_csv = data_final_csv.dropna(how="any", subset=[allele, peptide])
    log.info(
        "Length of dataframe AFTER removing duplicates and missing values: %s", len(data_final_csv)
    )
    data_final_csv.to_csv(output)
    log.info("Finished generation!")
