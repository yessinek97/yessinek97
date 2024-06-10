"""Module to process data and apply feature selection."""
import warnings
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd

from ig import CONFIGURATION_DIRECTORY, DATA_DIRECTORY, PROC_SEC_ID
from ig.utils.io import check_and_create_folder, load_yml, read_data, save_yml
from ig.utils.logger import get_logger, init_logger
from ig.utils.processing import (  # data_cleaning,; fix_expression_columns,; get_columns_with_unique_value,; get_data_type,; get_list_type,; merge_expression,; remove_columns,; remove_duplicate_columns,
    LoadProcessingConfiguration,
    add_features,
    cross_validation,
    data_processor_new_data,
    data_processor_single_data,
    find_duplicated_columns_by_values,
    get_columns_by_keywords,
    get_features_from_features_configuration,
    get_features_type,
    legend_replace_renamed_columns,
    load_dataset,
    report_missing_columns,
    union_type_from_features_type,
)

log: Logger = get_logger("Processing_main")
warnings.simplefilter(action="ignore")


@click.command()  # noqa
@click.option(
    "--train_path",
    "-t",
    type=str,
    default=None,
    help="Path to the train dataset.",
)
@click.option(
    "--main_data_path",
    "-mdp",
    type=str,
    multiple=True,
    default=None,
    help="Path to the dataset used for the main processing.",
)
@click.option(
    "--main_data_name",
    "-mdn",
    type=str,
    multiple=True,
    default=None,
    help="name  the dataset used for the main processing.",
)
@click.option(
    "--other_data_path",
    "-odp",
    type=str,
    multiple=True,
    default=None,
    help="Path to the other dataset ",
)
@click.option(
    "--other_data_name",
    "-odn",
    type=str,
    multiple=True,
    default=None,
    help="name the other dataset",
)
@click.option(
    "--configuration_path",
    "-c",
    type=str,
    default=None,
    help="name of configuration file under configuration diractory.",
)
@click.option(
    "--output_path",
    "-o",
    type=str,
    help="Output directory name",
    default=None,
)
@click.option(
    "--ignore_missing_features",
    "-ignore",
    is_flag=True,
    help="ignore missing features while processing other data ",
)
def processing(
    train_path: Optional[str],
    main_data_path: Optional[List[str]],
    main_data_name: Optional[List[str]],
    other_data_path: Optional[List[str]],
    other_data_name: Optional[List[str]],
    configuration_path: Optional[str],
    output_path: Optional[str],
    ignore_missing_features: bool,
) -> None:
    """Process the given raw data and save it with the features configuration file.

    Args:
    train_path :              train data path
    main_data_path :          list of principal data path
    other_data_path:          list of other data path
    main_data_name :          list of principal data name
    other_data_name:          list of other data name
    configuration_path :      processing configuration file name
    output_path :             output directory where the processed data and metadata are saved
    ignore_missing_features : ignore missing features while processing other data
    """
    if train_path:
        assert configuration_path

        process_main_data(train_path, main_data_path, main_data_name, configuration_path)
    if other_data_path:
        assert other_data_path
        assert other_data_name
        assert output_path
        output_dir = DATA_DIRECTORY / output_path
        init_logger(logging_directory=output_dir, file_name="other_data_proc")
        if len(other_data_path) == len(other_data_name):
            process_other_data(
                other_data_path, other_data_name, output_dir, ignore_missing_features
            )
        else:
            raise Exception("The length of other_data_path don't match the one of other_data_name")


def process_main_data(  # noqa
    train_path: str,
    main_data_path: Optional[List[str]],
    main_data_name: Optional[List[str]],
    configuration_path: str,
) -> None:
    """Process main data."""
    configuration = LoadProcessingConfiguration(CONFIGURATION_DIRECTORY / configuration_path)
    check_and_create_folder(configuration.output_dir)
    init_logger(logging_directory=configuration.output_dir, file_name="data_proc")
    features_tracker = {}
    data: Dict[str, pd.DataFrame] = {}
    processed_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    input_data_directory = Path(train_path).parent
    base_common_features: Optional[List[str]] = None

    data["train"] = load_dataset(train_path)
    if (main_data_path is not None) and (main_data_name is not None):

        if len(main_data_path) == len(main_data_name):
            for path, name in zip(main_data_path, main_data_name):
                data[name] = load_dataset(path)
        else:
            raise Exception("The length of main_data_path don't match the one of main_data_name")

    for data_name in data:
        initial_shape = data[data_name].shape
        log.info("-Start processing %s", data_name)
        processed_data[data_name] = data_processor_single_data(
            data[data_name].copy(), configuration
        )
        log.info("-Finish  processing %s ", data_name)
        log.info("-%s Before %s ", data_name, initial_shape)
        log.info("-%s After  %s ", data_name, processed_data[data_name]["proc"].shape)
        features_tracker[f"{data_name}_after_processing"] = processed_data[data_name][
            "proc"
        ].columns.tolist()
        log.info("#" * 50)
    log.info("#" * 50)

    if configuration.legend_path is not None:
        legend_path = input_data_directory / configuration.legend_path
        if legend_path.exists():
            legend = read_data(legend_path)
            base_common_features = legend[
                legend[configuration.legend_filter_column] == configuration.legend_value
            ][configuration.legend_feat_col_name].tolist()
            assert base_common_features
            base_common_features = [col.lower() for col in base_common_features]
            base_common_features = legend_replace_renamed_columns(
                base_common_features, configuration
            )
            log.info("- Features from legend  : %s", len(base_common_features))
        else:
            raise FileExistsError("legend file does not exist")

    common_features_set = set(processed_data["train"]["proc"].columns)
    for name in data:
        common_features_set = common_features_set & set(processed_data[name]["proc"].columns)
    log.info("- Common features between main data set : %s", len(common_features_set))
    if base_common_features:
        common_features_set = common_features_set & set(base_common_features)
        log.info(
            "- Common features between main data set and legend : %s", len(common_features_set)
        )
    common_features = list(common_features_set)

    log.info("#" * 50)

    duplicated_features = find_duplicated_columns_by_values(
        processed_data, common_features, configuration.include_features
    )
    common_features = [col for col in common_features if col not in duplicated_features]
    log.info(
        "- Common features between main data set after removing duplicated features: %s",
        len(common_features),
    )
    features_tracker["duplicated_features"] = duplicated_features
    log.info("#" * 50)
    log.info("- Exclude features :")
    exclude_features = []
    for keyword in configuration.exclude_features:
        single_exclude_features = get_columns_by_keywords(common_features, keyword)
        log.info("-- %s  : %s features", keyword, len(single_exclude_features))
        exclude_features.extend(single_exclude_features)
    log.info("-- All exclude features : %s", len(exclude_features))
    features_tracker["exclude_features"] = exclude_features
    common_features = [col for col in common_features if col not in exclude_features]
    log.info(
        "- Common features between main data set after removing exclude features: %s",
        len(common_features),
    )
    features_tracker["common_features"] = common_features
    log.info("#" * 50)
    log.info("- Separate feature per type(int,float,bool,object) ")
    features_type = get_features_type(
        {n: processed_data[n]["proc"] for n in processed_data}, common_features
    )

    int_features = union_type_from_features_type(features_type, int)
    float_features = union_type_from_features_type(features_type, float)
    bool_features = union_type_from_features_type(features_type, bool)
    object_features = union_type_from_features_type(features_type, object)
    log.info("--- int    : %s", len(int_features))
    log.info("--- float  : %s", len(float_features))
    log.info("--- bool   : %s", len(bool_features))
    log.info("--- object : %s", len(object_features))
    common_features = features_type.Name.tolist()
    log.info("- Common features between main data set with the same type: %s", len(common_features))

    log.info("#" * 50)
    log.info("- include features :")
    include_features = []
    for keyword in configuration.include_features:
        single_include_features = get_columns_by_keywords(common_features, keyword)
        log.info("-- %s  : %s features", keyword, len(single_include_features))
        include_features.extend(single_include_features)
    bnt_features = [col for col in common_features if col not in include_features]
    log.info("-- All include features : %s", len(include_features))
    log.info("-- base features :%s", len(bnt_features))
    features_tracker["include_features"] = include_features
    features_tracker["base_features"] = bnt_features

    log.info("#" * 50)
    log.info("- Generate features configuration file")
    final_common_features: List[str] = []
    features_configuration = {}
    features_configuration["id"] = configuration.id
    features_configuration["ids"] = [  # type: ignore
        col for col in configuration.base_columns if col != configuration.label
    ]
    features_configuration["label"] = configuration.label
    add_features(
        "float",
        features_configuration,
        float_features,
        include_features,
        bnt_features,
        configuration.features_float,
        configuration.features_include_float,
        final_common_features,
    )
    add_features(
        "int",
        features_configuration,
        int_features,
        include_features,
        bnt_features,
        configuration.features_int,
        configuration.features_include_int,
        final_common_features,
    )
    add_features(
        "bool",
        features_configuration,
        bool_features,
        include_features,
        bnt_features,
        configuration.features_bool,
        configuration.features_include_bool,
        final_common_features,
    )
    add_features(
        "categorical",
        features_configuration,
        object_features,
        include_features,
        bnt_features,
        configuration.features_categorical,
        configuration.features_include_categorical,
        final_common_features,
    )

    bnt_features = [col for col in final_common_features if col not in include_features]
    log.info("--Final common features between main data: %s", len(final_common_features))
    log.info("--- All include features : %s", len(include_features))
    log.info("--- base features : %s", len(bnt_features))
    columns_to_save = final_common_features + [PROC_SEC_ID]  # add fold

    if "train" in processed_data:
        if configuration.split:

            log.info("#" * 50)
            processed_data["train"]["base"], _ = cross_validation(
                processed_data["train"]["base"], configuration
            )
    log.info("#" * 50)
    log.info("- Save data")
    for name in processed_data:
        b_data = processed_data[name]["base"]
        p_data = processed_data[name]["proc"]
        f_data = b_data.merge(
            p_data[[col for col in p_data.columns if col in columns_to_save]],
            on=PROC_SEC_ID,
            how="left",
        )
        log.info(
            "-- %s with shape %s is saved under %s.csv",
            name,
            f_data.shape,
            configuration.output_dir / name,
        )
        f_data.to_csv(configuration.output_dir / f"{name}.csv", index=False)

    configuration.save_configuration()
    save_yml(features_configuration, configuration.output_dir / configuration.features_file_name)
    save_yml(features_tracker, configuration.output_dir / "features_tracker.yml")


def process_other_data(
    other_data_path: List[str],
    other_data_name: List[str],
    output_dir: Path,
    ignore_missing_features: bool,
) -> None:
    """Process other data."""
    data: Dict[str, pd.DataFrame] = {}
    configuration = LoadProcessingConfiguration(output_dir=output_dir)
    features_configuration_path = configuration.output_dir / configuration.features_file_name
    if not features_configuration_path.exists():
        raise FileExistsError(f"{features_configuration_path} is no available  ")

    features_configuration = load_yml(features_configuration_path)
    for path, name in zip(other_data_path, other_data_name):
        data[name] = load_dataset(path)
    features = get_features_from_features_configuration(features_configuration)

    for data_name in data:
        log.info("- Start processing %s", data_name)

        proc_data = data_processor_new_data(data[data_name].copy(), configuration)
        log.info("- Check features")
        # check_missing_columns(proc_data,features)
        log.info(
            "- Keep only %s features defined in the features configuration file", len(features)
        )
        report_missing_columns(proc_data, features, ignore_missing_features)
        proc_data = proc_data[[col for col in features if col in proc_data.columns] + [PROC_SEC_ID]]
        log.info(
            "- %s with shape %s is saved under %s.csv",
            data_name,
            proc_data.shape,
            configuration.output_dir / data_name,
        )
        proc_data.to_csv(configuration.output_dir / f"{data_name}.csv", index=False)
        log.info("#" * 50)
