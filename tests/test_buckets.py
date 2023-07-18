# type: ignore
"""This module includes the tests for buckets module."""
import os

import pytest
from click.testing import CliRunner
from google.cloud import storage

from ig.buckets import pull, push


def test_pull(bucket_path: str, local_path: str) -> None:
    """This function tests pull function."""
    runner = CliRunner()
    params = [
        "--bucket_path",
        bucket_path + "/test_data.tsv",
        "--local_path",
        local_path + "/test_data.tsv",
    ]
    _ = runner.invoke(pull, params)
    assert os.path.exists(local_path + "/test_data.tsv")


@pytest.mark.skip(reason="Testing function causes some credential issues with the CI pipeline")
def test_push(bucket_path: str, local_path: str) -> None:
    """This function tests pull function."""
    runner = CliRunner()
    bucket_path = bucket_path.replace("gs://biondeep-data/", "")
    params = ["--bucket_path", bucket_path, "--local_path", local_path + "/test_data_copy.tsv"]
    _ = runner.invoke(push, params)

    name = bucket_path + "/test_data_copy.tsv"
    storage_client = storage.Client()
    bucket_name = "biondeep-data"
    bucket = storage_client.bucket(bucket_name)
    stats = storage.Blob(bucket=bucket, name=name).exists(storage_client)
    assert stats is True
