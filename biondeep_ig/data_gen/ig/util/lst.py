"""Helper functions for list."""
import re
from typing import Any
from typing import Dict
from typing import Generator
from typing import List


def split_list(lst: List, n: int) -> Generator:
    """Yield successive n-sized chunks from lst.

    Args:
        lst: a list of elements
        n: length of chunk

    Returns:
        a generator
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def map_index_to_dict(input_dict: Dict) -> Dict[Any, int]:
    """Map sorted keys to indexes.

    The Keys in the dictionary "input_dict" must be sortable.

    Args:
        input_dict: a dictionary of elements

    Returns:
        indexed Dictionary
    """
    return dict(zip(sorted(input_dict.keys()), range(len(input_dict))))


def atoi(text: str) -> object:
    """Returns the digit format of the text."""
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[Any]:
    """Sorts in human order."""
    return [atoi(c) for c in re.split(r"(\d+)", text)]
