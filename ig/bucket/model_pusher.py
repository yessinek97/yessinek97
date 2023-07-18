# type: ignore
"""Define classes to handle the pushing of the different model types in the Google / AWS bucket."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from ig.bucket.click.constants import MODELS_BUCKET_NAME
from ig.bucket.gs import GSManager
from ig.src.logger import get_logger
from ig.src.models import BaseModel

log = get_logger("Model Pusher")


class BaseModelPusher(ABC):
    """Base abstract class to define interface for model pusher."""

    def __init__(self, use_s3: bool) -> None:
        """Instantiate model pusher."""
        self.bucket_manager = GSManager(MODELS_BUCKET_NAME)
        self.use_s3 = use_s3

    def push(self, model: BaseModel, **kwargs) -> None:
        """Main method to push the files linked to a model."""
        files_to_upload = self.get_files_to_upload(model, **kwargs)
        for fp in files_to_upload:
            self.bucket_manager.upload(local_file_path=fp, force=True)

    @abstractmethod
    def get_files_to_upload(self, model: BaseModel, **kwargs) -> Iterable[Path]:
        """Retrieve the list of files to upload.

        It must be implemented in child classes.
        """
