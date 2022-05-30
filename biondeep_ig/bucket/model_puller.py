"""Define classes to handle the pulling of the different model types in the Google / AWS bucket."""
from abc import ABC
from abc import abstractmethod
from typing import Iterable
from typing import Type

from cloudpathlib import CloudPath

from biondeep_ig.bucket.utils import get_bucket_manager
from biondeep_ig.src.logger import get_logger

log = get_logger("Puller")


class BaseModelPuller(ABC):
    """Base abstract class to define interface for model puller."""

    def pull(self, model_uri: str, **kwargs) -> None:
        """Main method to pull the files linked to a model."""
        bucket_manager = get_bucket_manager(model_uri, is_models_bucket=True)
        files_to_download = self.get_files_to_download(model_uri, **kwargs)
        for fp in files_to_download:
            bucket_manager.download(
                bucket_file_path=fp,
                force=True,
                # Mandatory since on s3 we do not have a dedicated bucket for models
                bucket_name=bucket_manager.default_bucket,
            )

    @abstractmethod
    def get_files_to_download(self, model_uri: str, **kwargs) -> Iterable[str]:
        """Retrieve the list of files to download.

        It must be implemented in child classes.
        """


class DefaultModelPuller(BaseModelPuller):
    """Default class to use to pull a model."""

    def get_files_to_download(self, model_uri: str, **kwargs) -> Iterable[str]:
        """Retrieve the list of files to download."""
        return {str(fp) for fp in CloudPath(model_uri).rglob("*") if fp.is_file()}


def get_model_puller(model_uri: str) -> BaseModelPuller:
    """Retrieve the model puller instance based on the type of the model.

    By default, the DefaultModelPuller is used.
    """
    puller_cls: Type[BaseModelPuller]

    puller_cls = DefaultModelPuller

    log.debug("%s used to pull the model %s", puller_cls.__name__, model_uri)

    return puller_cls()
