"""Main script."""
import click

from ig.buckets import pull, push
from ig.cimt import cimt, cimt_features_selection, cimt_inference, cimt_kfold_split, cimt_train
from ig.compute_metrics import compute_metrics
from ig.ensemble import ensemblexprs, ensoneexp
from ig.feature_selection import featureselection
from ig.gene_ontology_pipeline import gene_ontology_pipeline
from ig.generate_pairs_peptide_allele import generate_pairs
from ig.inference import exp_inference, inference
from ig.multi_trainer import multi_exp_inference, multi_inference, multi_train
from ig.multi_trainer_distributed import multi_train_distributed
from ig.processing import processing
from ig.run_esm_embeddings import compute_embeddings
from ig.trainer import compute_comparison_score, train, train_seed_fold, tune


@click.group()
def main() -> None:
    """Entry point for biondeep."""


main.add_command(gene_ontology_pipeline)
main.add_command(featureselection)
main.add_command(train)
main.add_command(train_seed_fold)
main.add_command(tune)
main.add_command(compute_metrics)
main.add_command(inference)
main.add_command(push)
main.add_command(pull)
main.add_command(ensemblexprs)
main.add_command(ensoneexp)
main.add_command(compute_comparison_score)
main.add_command(multi_train)
main.add_command(generate_pairs)
main.add_command(cimt_kfold_split)
main.add_command(cimt_features_selection)
main.add_command(cimt_train)
main.add_command(cimt)
main.add_command(cimt_inference)
main.add_command(processing)
main.add_command(multi_inference)
main.add_command(multi_train_distributed)
main.add_command(exp_inference)
main.add_command(multi_exp_inference)
main.add_command(compute_embeddings)

if __name__ == "__main__":
    main()
