"""Module used to define common constants used for command lines."""
from typing import Iterable
from typing import TypeVar


OneOrManyPathType = TypeVar("OneOrManyPathType", str, Iterable[str])
DATA_BUCKET_NAME = "biondeep-data"
MODELS_BUCKET_NAME = "biondeep-models"
GS_BUCKET_PREFIX = "gs://"
BUCKET_PREFIXES = GS_BUCKET_PREFIX
