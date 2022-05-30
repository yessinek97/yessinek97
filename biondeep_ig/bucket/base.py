"""Module to define the manager base class for upload/download operations on a bucket."""
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import click
from cloudpathlib import CloudPath
from cloudpathlib.exceptions import CloudPathNotADirectoryError

from biondeep_ig import DATA_DIRECTORY
from biondeep_ig import MODELS_DIRECTORY
from biondeep_ig.bucket.click import constants
from biondeep_ig.src.logger import get_logger

log = get_logger("Base")


class BaseBucketManager(ABC):
    """An abstract class to define the utility manager to upload / download files to / from a bucket."""

    def __init__(self, default_bucket: str, verbose: Union[int, bool] = True) -> None:
        """Initialize the manager.

        Args:
            default_bucket: name of the default bucket to use.
            verbose: indicates if the log should
            be displayed to the user regarding upload / download.
        """
        self.default_bucket = default_bucket
        self.verbose = verbose
        self.set_client()

    def delete(self, bucket_file_path: str, bucket_name: Optional[str] = None) -> None:
        """Delete a directory/file from the bucket.

        Args:
            bucket_file_path: path to the directory/file
            to delete on the bucket. The bucket prefix is optional.
            bucket_name: name of the bucket to use.
        """
        cloud_path = self.get_cloud_path(bucket_file_path, bucket_name)

        try:
            cloud_path.rmtree()
            self.log(
                f"The directory {cloud_path} has been removed on the {self.service_name} bucket."
            )
        except CloudPathNotADirectoryError:
            cloud_path.unlink()
            self.log(f"The file {cloud_path} has been removed on the {self.service_name} bucket.")

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Prefix of the bucket."""

    @property
    @abstractmethod
    def service_name(self) -> str:
        """Name of the bucket service."""

    @staticmethod
    @abstractmethod
    def set_client() -> None:
        """Set the bucket client."""

    def log(self, message: str):
        """Log a message if verbose.

        Args:
            message: message to log
        """
        if self.verbose:
            log.info(message)

    def extract_bucket_from_uri(self, uri: str) -> Optional[str]:
        """Extract the bucket name from the URI if possible."""
        if uri.startswith(self.prefix):
            _, _, bucket, *_ = uri.split("/")
            self.log(f"Bucket name {bucket} extracted from bucket_file_path")
        else:
            bucket = None

        return bucket

    def get_bucket_name(self, bucket_file_path: str, bucket_name: Optional[str] = None) -> str:
        """Get the bucket from the bucket file path and the optional bucket name.

        Remark: If bucket_name is not provided,
        it extracts the bucket from bucket_file_path or,
        if the latter is not an absolute path, it uses the default bucket.

        Args:
            bucket_file_path: path to the file/directory on the bucket
            bucket_name: name of the bucket to use

        Returns:
            the bucket name to use

        Raises:
            ValueError: if the bucket can be inferred
            from neither bucket_name, bucket_file_path or self.default_bucket
        """
        bucket_name = (
            bucket_name or self.extract_bucket_from_uri(bucket_file_path) or self.default_bucket
        )

        if not bucket_name:
            msg = """The bucket can not be inferred
            since bucket_name is not provided, the bucket name cannot be
            extracted from the bucket_file_path, and self.default_bucket is not set.
            """
            raise ValueError(msg)

        return bucket_name

    def get_cloud_path(self, bucket_file_path: str, bucket_name: Optional[str] = None) -> CloudPath:
        """Get the cloud path from the bucket file path and the optional bucket name.

        If bucket_name is given and the bucket in bucket_file_path
        is different, the latter is replaced by bucket_name.

        Args:
            bucket_file_path: path to the file/directory on the bucket
            bucket_name: name of the bucket to use

        Returns:
            the cloud path associated to the bucket file path and the optional bucket name
        """
        bucket_name = self.get_bucket_name(bucket_file_path, bucket_name)
        bucket_file_path = bucket_file_path.replace(f"{self.prefix}{bucket_name}/", "")

        if bucket_file_path.startswith(self.prefix):
            bucket_file_path = "/".join(bucket_file_path.split("/")[3:])

        return CloudPath(f"{self.prefix}{bucket_name}") / bucket_file_path

    def upload(
        self,
        local_file_path: Union[str, Path],
        bucket_file_path: Optional[str] = None,
        bucket_name: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Upload a local directory/file to the bucket.

        Remarks:
        - If the bucket_file_path is specified, the file is uploaded to this path.
        - If bucket_file_path is a bucket directory,
         the file is uploaded to this bucket with its original file name.
        - If bucket_file_path is not provided,
        the directory hierarchy is kept respective to models/ or data/ folders.


        Args:
            local_file_path: path to the local file or directory
            bucket_file_path: optional path to the bucket file/directory
            where to store the local file on the bucket.
            The prefix {self.prefix}/{bucket_name} is optional.
            bucket_name: optional name of the bucket to use.
            Otherwise inferred from bucket_file_path or equal to self.default_bucket.
            force: indicates if the bucket path should be overwritten if it already exists.

        Returns: URI of the file / directory uploaded.

        Raises:
            FileNotFoundError: if the file does not exist locally.
            ValueError: if bucket_file_path
            not specified and bucket different than models' or data' buckets
        """
        local_file_path = Path(local_file_path).resolve()

        if not local_file_path.exists():
            raise FileNotFoundError(f"The file/directory {local_file_path} does not exist locally.")

        if bucket_file_path is None:
            bucket_name = bucket_name or self.default_bucket

            if bucket_name in {constants.MODELS_BUCKET_NAME}:
                bucket_file_path = str(local_file_path.relative_to(MODELS_DIRECTORY))

            elif bucket_name == constants.DATA_BUCKET_NAME:
                bucket_file_path = str(local_file_path.relative_to(DATA_DIRECTORY))

            else:
                msg = (
                    f"If the bucket_file_path parameter is not specified, you have to use "
                    f"{constants.MODELS_BUCKET_NAME} "
                    f"or {constants.DATA_BUCKET_NAME} buckets. \
                     Bucket used currently: {bucket_name}."
                )
                raise ValueError(msg)

        # in case we upload a file to a directory, we need to add the file name because
        # CloudPath does not know this is a directory if the bucket does not exist yet
        if local_file_path.is_file() and Path(bucket_file_path).suffix == "":
            bucket_file_path = str(Path(bucket_file_path) / local_file_path.name)

        cloud_path = self.get_cloud_path(bucket_file_path, bucket_name)

        should_upload = (
            force
            or not cloud_path.exists()
            or (
                cloud_path.is_dir()
                and click.confirm(
                    f"The directory {cloud_path} already exists in {self.service_name} \
                    bucket, do you want to overwrite it?"
                )
            )
            or (
                cloud_path.is_file()
                and click.confirm(
                    f"The file {cloud_path} already exists in {self.service_name} \
                     bucket, do you want to overwrite it?"
                )
            )
        )

        if should_upload:
            cloud_path = cloud_path.upload_from(local_file_path, force_overwrite_to_cloud=True)
            self.log(
                f"{local_file_path} uploaded to the {self.service_name} bucket at {cloud_path}"
            )

        return cloud_path.as_uri()

    def upload_files(
        self,
        file_paths: List[str],
        bucket_folder: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ) -> List[str]:
        """Upload multiple files to the bucket.

        Args:
            file_paths: List of files to upload
            bucket_folder: optional folder to use
            to upload the file in the bucket.
            The prefix {self.prefix}/{bucket_name} is optional.
            bucket_name: optional name of the bucket to use.

        Returns:
            paths of the files in the bucket.
        """
        uploaded_paths = [
            self.upload(
                local_file_path=fp,
                bucket_file_path=bucket_folder,
                bucket_name=bucket_name,
            )
            for fp in file_paths
        ]

        return uploaded_paths

    def download(
        self,
        bucket_file_path: str,
        local_file_path: Optional[Union[Path, str]] = None,
        bucket_name: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Download a file/directory from the bucket to the local machine.

        If local_file_path is specified, the file is downloaded to this file path,
        otherwise the structure of the file path is preserved.

        Args:
            bucket_file_path: path to the file to download.
            The prefix {self.prefix}{bucket_name} is optional.
            local_file_path: optional directory or full path to use to download the file.
                - if this is a directory, the original filename is kept
                - if this is a full path, the filename is overwritten with the one provided
            bucket_name: optional name of the bucket to use.
            Otherwise inferred from bucket_file_path or equal to self.default_bucket.
            force: indicates if the local path should be overwritten if it already exists.

        Returns:
            path of the file saved locally

        Raises:
            FileNotFoundError: if file does not exist in the bucket
            ValueError: if local_file_path not
            specified and bucket different than models' or data' buckets
        """
        cloud_path = self.get_cloud_path(bucket_file_path, bucket_name)

        if not cloud_path.exists():
            raise FileNotFoundError(
                f"The file/directory {bucket_file_path} does not \
                exist in {self.service_name} bucket '{bucket_name}'"
            )

        if local_file_path is None:
            bucket_name = self.get_bucket_name(bucket_file_path, bucket_name)
            bucket_file_path = bucket_file_path.replace(self.prefix, "").replace(
                f"{bucket_name}/", ""
            )

            if bucket_name in {constants.MODELS_BUCKET_NAME}:
                local_file_path = MODELS_DIRECTORY / bucket_file_path

            elif bucket_name == constants.DATA_BUCKET_NAME:
                local_file_path = DATA_DIRECTORY / bucket_file_path

            else:
                msg = (
                    f"If the local_file_path is not \
                     specified, you have to use {constants.MODELS_BUCKET_NAME} "
                    f"or {constants.DATA_BUCKET_NAME} buckets. "
                    f"Bucket used currently: {bucket_name}."
                )
                raise ValueError(msg)
        else:
            # resolve file path to handle relative_to()
            local_file_path = Path(local_file_path).resolve()

        should_download = (
            force
            or not local_file_path.exists()
            or (
                local_file_path.is_dir()
                and click.confirm(
                    f"The directory {local_file_path} already exists locally, "
                    f"do you want to download and overwrite it?"
                )
            )
            or (
                local_file_path.is_file()
                and click.confirm(
                    f"The file {local_file_path} already exists locally, "
                    f"do you want to download and overwrite it?"
                )
            )
        )

        if should_download:
            # Due to a bug with cloudpathlib (see https://github.com/drivendataorg/cloudpathlib/issues/57),
            # we ensure that the parent directory of local_file_path (or itself) is created
            parent_directory = local_file_path.parent if local_file_path.suffix else local_file_path
            parent_directory.mkdir(exist_ok=True, parents=True)

            cloud_path.download_to(local_file_path)
            self.log(
                f"{cloud_path} downloaded from the {self.service_name} bucket at {local_file_path}"
            )

        return str(local_file_path)
