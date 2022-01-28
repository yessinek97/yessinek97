"""Setup to define the repository as a package."""
from codecs import open
from typing import List

import setuptools
from setuptools import setup

__version__ = "0.1.0"


def read_requirements(file: str) -> List[str]:
    """Returns content of given requirements file."""
    return [line for line in open(file) if not (line.startswith("#") or line.startswith("--"))]


setup(
    name="biondeep_ig",
    version=__version__,
    author="InstaDeep",
    url="https://gitlab.com/instadeep/biondeep-ig",
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=read_requirements("./requirements.txt"),
    include_package_data=True,
)
