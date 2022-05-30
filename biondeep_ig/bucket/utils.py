"""Module to define utils functions regarding bucket operations."""
from pathlib import Path
from typing import Optional

from biondeep_ig import DATA_DIRECTORY
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig.bucket.base import BaseBucketManager
from biondeep_ig.bucket.click.constants import BUCKET_PREFIXES
from biondeep_ig.bucket.click.constants import DATA_BUCKET_NAME
from biondeep_ig.bucket.click.constants import GS_BUCKET_PREFIX
from biondeep_ig.bucket.click.constants import MODELS_BUCKET_NAME
from biondeep_ig.bucket.gs import GSManager


def get_bucket_manager(
    bucket_path: Optional[str], is_models_bucket: bool = False, **kwargs
) -> BaseBucketManager:
    """Get bucket utility manager.

    We use by default the GSManager.

    Args:
        path: path to identify the manager class
        is_models_bucket: whether the bucket is for models' storage
        kwargs: keyword arguments forwarded to the bucket manager

    Returns:
        bucket utility manager
    """
    if bucket_path is None:
        print("Path is Empty please make sure to add the path.")
    return GSManager(MODELS_BUCKET_NAME if is_models_bucket else None, **kwargs)


def remove_bucket_prefix(file_path: str) -> str:
    """Remove the bucket prefix and the bucket name from the file path.

    Args:
        file_path: file path to clean

    >>> remove_bucket_prefix("gs://biondeep-models/presentation/first_model")
    'presentation/first_model'

    >>> remove_bucket_prefix("gs://random-bucket/directory/test.csv")
    'directory/test.csv'
    """
    return str(Path(*Path(file_path.replace(GS_BUCKET_PREFIX, "")).parts[1:]))


def get_local_path(bucket_path: str) -> str:
    """Get the local path corresponding to the input GS path.

    - for bucket 'biondeep-models': use MODELS_DIRECTORY as prefix
    - for bucket 'biondeep-data': use DATA_DIRECTORY as prefix

    Raises:
        ValueError: if the bucket_path does not start by GS_BUCKET_PREFIX or the bucket used
            is not known (i.e. not DATA_BUCKET_NAME or MODELS_BUCKET_NAME)
    """
    if not bucket_path.startswith(BUCKET_PREFIXES):
        raise ValueError(
            f"The bucket path ({bucket_path}) provided does not start with a known bucket prefix."
            f"Currently known bucket prefix are '{GS_BUCKET_PREFIX}'"
        )

    local_path = remove_bucket_prefix(bucket_path)

    if DATA_BUCKET_NAME in bucket_path:
        local_path = str(DATA_DIRECTORY / local_path)

    elif MODELS_BUCKET_NAME in bucket_path:
        local_path = str(MODELS_DIRECTORY / local_path)

    else:
        raise ValueError(
            f"The bucket used is unknown. Currently \
             known buckets are '{DATA_BUCKET_NAME}' and '{MODELS_BUCKET_NAME}'"
        )

    return local_path
