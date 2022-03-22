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
    entry_points={
        "console_scripts": [
            # data generation
            "agg_scores=biondeep_ig.data_gen.ig.data_gen.parse_score_files:main",
            "tcr_gen=biondeep_ig.data_gen.ig.data_gen.tcr.generate_tcrs:main",
            "pmhc_gen=biondeep_ig.data_gen.ig.data_gen.pmhc.generate_pmhc:main",
            "tcr_pmhc_extract=biondeep_ig.data_gen.ig.data_gen.tcr_pmhc.tcr_pmhc_extract:main",
            "tcr_pmhc_align=biondeep_ig.data_gen.ig.data_gen.tcr_pmhc.tcr_pmhc_align:main",
            "train=biondeep_ig.trainer:train",
            "train-seed-fold=biondeep_ig.trainer:train_seed_fold",
            "tune=biondeep_ig.trainer:tune",
            "featureselection=biondeep_ig.feature_selection:featureselection",
            "inference=biondeep_ig.inference:inference",
            "compute-metrics=biondeep_ig.compute_metrics:compute_metrics",
        ]
    },
)
