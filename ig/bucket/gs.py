# type: ignore
"""Module to define the manager handling basic upload/download operations on a gs bucket (GCP)."""
import os
from typing import Optional, Union

from cloudpathlib import GSClient

from ig.bucket.base import BaseBucketManager
from ig.bucket.click import constants
from ig.src.logger import get_logger

log = get_logger("Pusher")


class GSManager(BaseBucketManager):
    """A GS utility manager to upload / download files to / from gs."""

    prefix = constants.GS_BUCKET_PREFIX
    service_name = "Google Storage"

    def __init__(
        self,
        default_bucket: Optional[str] = None,
        verbose: Union[int, bool] = True,
    ) -> None:
        """Initialize the GS client.

        Args:
            default_bucket: name of the default bucket to use.
            verbose: indicates if the log should be displayed
            to the user regarding upload / download.
        """
        super().__init__(
            default_bucket or os.getenv("GS_BUCKET_NAME", constants.DATA_BUCKET_NAME), verbose
        )

    @staticmethod
    def set_client() -> None:
        """Set the GS client."""
        GSClient().set_as_default_client()
