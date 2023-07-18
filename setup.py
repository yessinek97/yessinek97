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
            # ablation study
            "ablation_study=ig.ablation_study:main",
            "ablation_post=ig.ablation_study:postprocess",
            # data generation
            "gene-ontology-pipeline=ig.gene_ontology_pipeline:gene_ontology_pipeline",
            # training
            "train=ig.trainer:train",
            "train-seed-fold=ig.trainer:train_seed_fold",
            "tune=ig.trainer:tune",
            "featureselection=ig.feature_selection:featureselection",
            "multi-train=ig.multi_trainer:multi_train",
            # inference
            "inference=ig.inference:inference",
            "exp-inference=ig.inference:exp_inference",
            "multi-inference=ig.multi_trainer:multi_inference",
            "multi-exp-inference=ig.multi_trainer:multi_exp_inference",
            "compute-metrics=ig.compute_metrics:compute_metrics",
            "ensemblexprs=ig.ensemble:ensemblexprs",
            "ensoneexp=ig.ensemble:ensoneexp",
            # push pull GCP command
            "push=ig.buckets:push",
            "pull=ig.buckets:pull",
            "compute_comparison_score=ig.trainer:compute_comparison_score",
            # Generate peptide allele pairs
            "generate-pairs=ig.generate_pairs_peptide_allele:generate_pairs",
            # cimt model
            "cimt-kfold-split=ig.cimt:cimt_kfold_split",
            "cimt-features-selection=ig.cimt:cimt_features_selection",
            "cimt-train=ig.cimt:cimt_train",
            "cimt=ig.cimt:cimt",
            "cimt-inference=ig.cimt:cimt_inference",
            # processing
            "processing=ig.processing:processing",
        ]
    },
)
