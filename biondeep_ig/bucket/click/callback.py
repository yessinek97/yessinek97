"""Define callacks for click arguments."""
from typing import Callable


def format_callback(f: Callable) -> Callable:
    """Function to wrap a function to use as a click callback.

    Taken from https://stackoverflow.com/a/42110044/8056572
    """
    return lambda _, __, x: f(x)
