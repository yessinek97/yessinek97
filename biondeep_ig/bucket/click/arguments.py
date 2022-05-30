"""Module used to define common arguments used for command lines.

Once defined in this module, an argument (or several arguments) can be
used as follows:

from biondeep.utils.click import arguments

....

@arguments.{argument_name}
def my_cli_command(...)
"""
import functools
from typing import Callable
from typing import Optional

import click

from biondeep_ig.bucket.click.constants import DATA_BUCKET_NAME
from biondeep_ig.bucket.click.constants import GS_BUCKET_PREFIX
from biondeep_ig.bucket.click.constants import MODELS_BUCKET_NAME
from biondeep_ig.bucket.click.custom_types import BucketOrDiskPath


def dataset_path(func):
    """Decorator used to define the dataset path argument."""

    @click.option(
        "--dataset_path",
        "-d",
        required=True,
        multiple=True,
        type=BucketOrDiskPath(exists=True),
        help=(
            "Path of the input dataset. It can be a local path or a bucket path "
            f"(should start with '{GS_BUCKET_PREFIX}{DATA_BUCKET_NAME}/'"
            "Multiple paths -d {dataset_path1} -d {dataset_path2} can be specified."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


def output_file_path(
    option_long_name: str,
    option_short_name: str,
    func: Optional[Callable] = None,
    required: bool = False,
    **kwargs,
):
    """Decorator used to define the output file path argument.

    The 'help' section of this argument can be
    overwritten by calling @output_file_path(..., help="My custom help").
    """

    def validate_length(ctx, param, value):
        # pylint: disable=W0621
        try:
            dataset_path = ctx.params["dataset_path"]
        except KeyError:
            raise click.BadOptionUsage(
                option_long_name, f"--dataset_path must be provided before {option_long_name}", ctx
            )
        msg = (
            f"- {len(dataset_path)} dataset paths provided: {' | '.join(dataset_path)} "
            f"- {len(value)} output files: {' | '.join(value)} "
        )

        if param.required and len(value) != len(dataset_path):
            msg = "There should be one output file path for every dataset path. " + msg
            raise ValueError(msg)
        if not param.required and len(value) > len(dataset_path):
            msg = "You have specified more output files than dataset paths. " + msg
            raise ValueError(msg)
        return value

    kwargs["help"] = kwargs.get(
        "help",
        (
            "Path of the file where the output will be saved. "
            f"If it starts by '{GS_BUCKET_PREFIX}{DATA_BUCKET_NAME} /', "
            "it will be pushed to the GS/S3 bucket.  \
            If not provided, the output columns are added to --dataset_path. "
            f"Multiple paths {option_short_name}"
            " {"
            f"{option_long_name.strip('-')}1"
            "} "
            f"{option_short_name}"
            " {"
            f"{option_long_name.strip('-')}2"
            "} can be specified."
        ),
    )

    def _outer_wrapper(func):
        @click.option(
            option_long_name,
            option_short_name,
            required=required,
            multiple=True,
            type=BucketOrDiskPath(exists=False),
            callback=validate_length,
            **kwargs,
        )
        @functools.wraps(func)
        def _inner_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return _inner_wrapper

    if func is None:
        return _outer_wrapper

    return _outer_wrapper(func)


def model_name(func: Optional[Callable] = None, multiple: bool = False):
    """Decorator used to define the model name argument.

    Args:
        func: function to wrap
        multiple: whether multiple 'model_name' arguments can be specified
    """

    def _outer_wrapper(func):
        @click.option(
            "--model_name",
            "-n",
            type=str,
            multiple=multiple,
            required=True,
            help="Model to use: it can be the name of the folder \
             where the model is saved, i.e. (experiment_name + date) or the path.",
        )
        @functools.wraps(func)
        def _inner_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return _inner_wrapper

    if func is None:
        return _outer_wrapper

    return _outer_wrapper(func)


def training_path(func):
    """Decorator used to define the training file path to use."""

    @click.option(
        "--training_path",
        "-t",
        type=BucketOrDiskPath(exists=True),
        required=True,
        help="Path to the training file.",
    )
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


def nb_gpus(func):
    """Decorator used to define the number of gpus to use."""
    # pylint: disable=W0613
    def validate_non_negative(ctx, param, value):
        if value is not None and value < 0:
            raise click.BadParameter("nb_gpus should be greater or equal to 0.")
        return value

    @click.option(
        "--nb_gpus",
        "-g",
        type=int,
        required=False,
        help=(
            "Number of GPUs to use. For commands using models  \
            which are already trained, if this parameter "
            "is not provided, the number of GPUs set during training will be used instead."
        ),
        callback=validate_non_negative,
    )
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper


def force(func=None, **kwargs):
    """Decorator used to define the force flag.

    The 'help' parameter for this parameter
    can be overwritten by calling @force(help="My custom help").
    """
    kwargs["help"] = kwargs.get(
        "help",
        "Skip prompts in the CLI command and overwrite \
        any existing columns, datasets or models (both locally or in "
        "the cloud) used by the command. \
        This parameter should be used with caution.",
    )

    def _outer_wrapper(func):
        @click.option("--force", "-f", is_flag=True, **kwargs)
        @functools.wraps(func)
        def _inner_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return _inner_wrapper

    if func is None:
        return _outer_wrapper

    return _outer_wrapper(func)


def push_to_bucket(func=None, **kwargs):
    """Decorator used to define the flag to indicate.

    if we should push artifacts to the bucket at the end of a command.

    By default the 'help' corresponds to an usage of this parameter with model training
    but it can be overwritten by calling @push_to_bucket(help="My custom help")
    """
    kwargs["help"] = kwargs.get(
        "help",
        f"If specified, push the trained model \
        to the bucket '{MODELS_BUCKET_NAME}' (with its best checkpoint "
        f"in case of a DL model) in the \
        folder biondeep-models/[model_type]/[model_name]/. This is done "
        f"automatically if the training is launched \
        on InstaDeep's infrastructure.",
    )

    def _outer_wrapper(func):
        @click.option(
            "--push_to_bucket",
            "-b",
            is_flag=True,
            default=False,
            **kwargs,
        )
        @functools.wraps(func)
        def _inner_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return _inner_wrapper

    if func is None:
        return _outer_wrapper

    return _outer_wrapper(func)


def use_s3(func):
    """Decorator used to define the flag to indicate.

    if the bucket service to upload/download content is AWS S3.
    """

    @click.option(
        "--use_s3",
        "-s3",
        required=False,
        is_flag=True,
        help="Use the AWS S3 bucket instead of the  \
        Google Storage bucket when pushing/pulling models.",
    )
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper
