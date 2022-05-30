"""Module used to define common custom types used for command lines."""
from typing import Optional

import click

from biondeep_ig.bucket.utils import get_bucket_manager


class BucketOrDiskPath(click.Path):
    """A path class that supports path for GS/S3 storage.

    The path get checked if all following conditions are met:
    - path is a GS/S3 path
    - `exists` is True, which means the file has to exist
    """

    name = "bucket_or_disk_path"

    def convert(
        self, value: str, param: Optional[click.core.Parameter], ctx: Optional[click.core.Context]
    ):
        """Check and convert the input value."""
        if not isinstance(value, str):
            raise ValueError(f"Value {value} is not string.")

        bucket_manager = get_bucket_manager(value, verbose=False)

        if not (self.exists and value.startswith(bucket_manager.prefix)):
            return super().convert(value, param, ctx)

        cloud_path = bucket_manager.get_cloud_path(value)
        if not cloud_path.exists():
            self.fail(f"{cloud_path} does not exist in {bucket_manager.service_name}.")

        return value


class BucketPath(click.ParamType):
    """Class to support bucket (GS or AWS S3) path."""

    name = "bucket_path"

    def __init__(self, is_models_bucket: bool = False, exists: bool = False) -> None:
        """Initialize the bucket path."""
        self.is_models_bucket = is_models_bucket
        self.exists = exists

    def convert(
        self, value: str, param: Optional[click.core.Parameter], ctx: Optional[click.core.Context]
    ):  # pylint: disable=W0613
        """Check and convert the input value."""
        if not self.exists:
            return value

        bucket_manager = get_bucket_manager(value, self.is_models_bucket, verbose=False)
        cloud_path = bucket_manager.get_cloud_path(value)

        if not cloud_path.exists():
            self.fail(f"{cloud_path} does not exist in {bucket_manager.service_name}.")

        return value
