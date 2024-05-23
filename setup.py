"""Setup to define the repository as a package."""
from codecs import open  # pylint: disable=redefined-builtin
from typing import List

import setuptools
from setuptools import setup

__version__ = "0.1.0"


def read_requirements(file: str) -> List[str]:
    """Returns content of given requirements file."""
    return [line for line in open(file) if not (line.startswith("#") or line.startswith("--"))]


setup(
    name="ig",
    version=__version__,
    author="InstaDeep",
    url="https://gitlab.com/instadeep/biondeep-ig",
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=read_requirements("./requirements.txt"),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            # training
            "train=ig.cli.trainer:train",
            "train-seed-fold=ig.cli.trainer:train_seed_fold",
            "tune=ig.cli.trainer:tune",
            "featureselection=ig.cli.feature_selection:featureselection",
            "multi-train=ig.cli.multi_trainer:multi_train",
            "predict-llm=ig.cli.llm_based_predictor:predict",
            # inference
            "inference=ig.cli.inference:inference",
            "exp-inference=ig.cli.inference:exp_inference",
            "multi-inference=ig.cli.multi_trainer:multi_inference",
            "multi-exp-inference=ig.cli.multi_trainer:multi_exp_inference",
            "compute-metrics=ig.cli.compute_metrics:compute_metrics",
            "ensemblexprs=ig.cli.ensemble:ensemblexprs",
            "ensoneexp=ig.cli.ensemble:ensoneexp",
            # push pull GCP command
            "push=ig.cli.buckets:push",
            "pull=ig.cli.buckets:pull",
            "compute_comparison_score=ig.cli.trainer:compute_comparison_score",
            # Generate peptide allele pairs
            "generate-pairs=ig.cli.generate_pairs_peptide_allele:generate_pairs",
            # cimt model
            "cimt=ig.cli.cimt:cimt",
            "cimt-kfold-split=ig.cli.cimt:cimt_kfold_split",
            "cimt-features-selection=ig.cli.cimt:cimt_features_selection",
            "cimt-train=ig.cli.cimt:cimt_train",
            "cimt-inference=ig.cli.cimt:cimt_inference",
            # processing
            "processing=ig.cli.processing:processing",
            "compute-embeddings=ig.cli.compute_embedding:compute_embedding",
        ]
    },
)
