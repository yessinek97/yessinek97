"""Main scirpt."""
import click

from biondeep_ig.compute_metrics import compute_metrics
from biondeep_ig.feature_selection import featureselection
from biondeep_ig.modular import modulartrain
from biondeep_ig.trainer import train
from biondeep_ig.trainer import train_seed_fold
from biondeep_ig.trainer import tune


@click.group()
def main():
    """Entry point for biondeep."""
    pass


main.add_command(featureselection)
main.add_command(modulartrain)
main.add_command(train)
main.add_command(train_seed_fold)
main.add_command(tune)
main.add_command(compute_metrics)

if __name__ == "__main__":
    main()
