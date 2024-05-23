"""Main script."""
import click

from ig.cli.buckets import pull, push
from ig.cli.cimt import cimt
from ig.cli.compute_embedding import compute_embedding
from ig.cli.compute_metrics import compute_metrics
from ig.cli.ensemble import ensemblexprs, ensoneexp
from ig.cli.feature_selection import featureselection
from ig.cli.generate_pairs_peptide_allele import generate_pairs
from ig.cli.inference import exp_inference, inference
from ig.cli.multi_trainer import multi_exp_inference, multi_inference, multi_train
from ig.cli.multi_trainer_distributed import multi_train_distributed
from ig.cli.processing import processing
from ig.cli.trainer import compute_comparison_score, train, train_seed_fold, tune


@click.group()
def main() -> None:
    """Entry point for biondeep."""


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
main.add_command(cimt)
main.add_command(processing)
main.add_command(multi_inference)
main.add_command(multi_train_distributed)
main.add_command(exp_inference)
main.add_command(multi_exp_inference)
main.add_command(compute_embedding)


if __name__ == "__main__":
    main()
