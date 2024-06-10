"""Module used to define some helper functions."""
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ig import DEFAULT_SEED, FS_CONFIGURATION_DIRECTORY, MAX_RANDOM_SEED
from ig.utils.io import load_yml
from ig.utils.logger import get_logger

log = get_logger("utils")


def load_fs(config: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load the available fs methods from the configuration file."""
    fs = config["FS"]["FS_methods"]
    fs_types = []
    fs_config = []
    for config_path in fs:
        fs = load_yml(FS_CONFIGURATION_DIRECTORY / "FS_method_config" / config_path)
        fs_types.append(fs["FS_type"])
        fs_config.append(fs["FS_config"])
    return fs_types, fs_config


def seed_basic(seed: int) -> None:
    """Seed every thing."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def remove_bucket_prefix(uri: str, keep: Optional[int] = -2) -> str:
    """Remove the bucket prefix and the bucket name from the file path in order to keep the relative path.

    Args:
        uri: uri to clean
        keep: int
    >>> remove_bucket_prefix("gs://biondeep-data/IG/data/Ig_2022/train.csv",-2)
    'Ig_2022/train.csv'
    >>> remove_bucket_prefix("gs://biondeep-data/IG/data/Ig_2022/train.csv",-1)
    'train.csv'
    """
    return os.path.join(*uri.split("/")[keep:])


def generate_random_seeds(nbr_seeds: int) -> List[int]:
    """Generate a list of random values between 0 and MAX_RANDOM_SEED with length equal to nbr_seeds."""
    random_seeds: List[int] = []
    random.seed(DEFAULT_SEED)
    while len(random_seeds) < nbr_seeds:
        random_seeds = list(set(random_seeds + [random.randint(0, MAX_RANDOM_SEED)]))
    return random_seeds


def crop_sequences(
    sequences: List[str], mutation_start_positions: List[int], context_length: int
) -> List[str]:
    """Crops the sequences to match desired context length."""
    sequences = [
        seq[
            max(0, mut_pos - context_length // 2) : max(0, mut_pos - context_length // 2)
            + context_length
        ]
        for seq, mut_pos in zip(sequences, mutation_start_positions)
    ]
    return sequences


def get_features_paths(configuration: Dict[str, Any]) -> List[str]:
    """Get the list of features paths."""
    features_paths = configuration.get("feature_paths", [])
    if features_paths:
        return features_paths
    return []
