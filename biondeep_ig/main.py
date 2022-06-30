"""Main scirpt."""
import click

from biondeep_ig.buckets import pull
from biondeep_ig.buckets import push
from biondeep_ig.compute_metrics import compute_metrics
from biondeep_ig.ensemble import ensemblexprs
from biondeep_ig.ensemble import ensoneexp
from biondeep_ig.feature_selection import featureselection
from biondeep_ig.inference import inference
from biondeep_ig.modular import modulartrain
from biondeep_ig.trainer import train
from biondeep_ig.trainer import train_seed_fold
from biondeep_ig.trainer import tune


@click.group()
def main():
    """Entry point for biondeep."""


main.add_command(featureselection)
main.add_command(modulartrain)
main.add_command(train)
main.add_command(train_seed_fold)
main.add_command(tune)
main.add_command(compute_metrics)
main.add_command(inference)
main.add_command(push)
main.add_command(pull)
main.add_command(ensemblexprs)
main.add_command(ensoneexp)

if __name__ == "__main__":
    main()
