"""Fixtures and variable used to test different codes in the project."""
# pylint: disable=redefined-outer-name
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def current_path():
    """Get Current Path to access the toy dataset."""
    return Path(__file__).resolve().parent
