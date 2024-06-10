"""Command lines to pull / push data from / to bucket (GCP or S3)."""
from logging import Logger

import click

from ig import DATA_DIRECTORY
from ig.bucket.click import arguments, custom_types
from ig.bucket.click.constants import (
    BUCKET_PREFIXES,
    DATA_BUCKET_NAME,
    GS_BUCKET_PREFIX,
    OneOrManyPathType,
)
from ig.bucket.utils import get_bucket_manager, get_local_path  # type: ignore
from ig.utils.logger import get_logger

log: Logger = get_logger("bucket")


@click.command()
@click.option(
    "--bucket_path",
    "-b",
    required=True,
    type=custom_types.BucketPath(exists=True),  # type: ignore
    help=(
        "Path of the dataset to pull from a GS/S3 bucket. "
        f"The bucket prefix is optional: by default  \
        it is set to '{GS_BUCKET_PREFIX}{DATA_BUCKET_NAME}/'."
    ),
)
@click.option(
    "--local_path",
    "-l",
    required=False,
    type=click.Path(exists=False),
    help=(
        "Local path to use to download the data. "
        f"If not specified the directory structure \
         of the bucket is kept relatively to {DATA_DIRECTORY}."
    ),
)
@arguments.force(help="Force overwrite the local file if it already exists.")  # type: ignore
def pull(bucket_path: str, local_path: str, force: bool) -> None:
    """Command line to download file or.

    directory from a GS/S3 bucket and save it locally.

    Args:
        bucket_path: path to the bucket
        local_path: the local path
        force: force to update the folder if any in the gcp.

    Returns:
        a bucket manager
    """
    bucket_manager = get_bucket_manager(bucket_path)
    bucket_manager.download(bucket_file_path=bucket_path, local_file_path=local_path, force=force)


@click.command()
@click.option(
    "--local_path",
    "-l",
    required=True,
    type=click.Path(exists=True),
    help="Path to the local file.",
)
@click.option(
    "--bucket_path",
    "-b",
    required=False,
    type=str,
    help=(
        "Path to the file in the bucket. \
        If not specified, the local directory structure is kept."
        f"The bucket prefix is optional: \
        by default it is set to '{GS_BUCKET_PREFIX}{DATA_BUCKET_NAME}/'."
    ),
)
@arguments.force(help="Force overwrite the bucket file if it already exists.")  # type: ignore
def push(local_path: str, bucket_path: str, force: bool) -> None:
    """Command line to upload local file / directory to a GS/S3 bucket."""
    bucket_manager = get_bucket_manager(bucket_path)
    bucket_manager.upload(local_file_path=local_path, bucket_file_path=bucket_path, force=force)


def maybe_pull_data(ctx: click.Context, path: OneOrManyPathType, force: bool) -> OneOrManyPathType:
    """Check if one or many datasets should be pulled.

    from a GS/S3 bucket and return local path(s).

    Args:
        ctx: click context
        path: path(s) of the file(s) that should be potentially pulled
        force: whether to overwrite data in case it already exists locally

    Returns:
        local path(s) of the file(s)

    """
    if isinstance(path, str):
        if path.startswith(BUCKET_PREFIXES):
            ctx.invoke(pull, bucket_path=path, force=force)
            return get_local_path(path)
        return path

    to_download = [p for p in path if p.startswith(BUCKET_PREFIXES)]
    path = [p for p in path if not p.startswith(BUCKET_PREFIXES)]
    for bucket_path in to_download:
        ctx.invoke(pull, bucket_path=bucket_path, force=force)
        path.append(get_local_path(bucket_path))

    return path
