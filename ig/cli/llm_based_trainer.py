"""Trainer used to train llm based models."""


from pathlib import Path

import click
import pandas as pd
from transformers import AutoTokenizer

from ig import CONFIGURATION_DIRECTORY
from ig.src.models import LLMBasedModel
from ig.src.utils import load_yml


@click.command()
@click.option(
    "--folder-name",
    "-n",
    type=str,
    required=True,
    help="name of the folder where the experiment output should be saved",
)
@click.option(
    "--configuration-file",
    "-c",
    type=str,
    required=True,
    help=" Path to configuration file.",
)
def train(configuration_file: str, folder_name: str) -> None:
    """Mehtod to train the chosen model on the IG prediction task.

    Args:
        configuration_file (str): Name of the config file to be used.
        folder_name (str): Name of the folder where experiment output will be saved.
    """
    # read configuration
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    llm_hf_model_path = general_configuration["model_architecture"]["llm_hf_model_path"]
    experiment_name = general_configuration["general_parameters"]["experiment_name"]
    dataset_name = general_configuration["general_parameters"]["dataset_name"]
    model_type = general_configuration["model_architecture"]["model_type"]

    # read data
    train_dataframe = pd.read_csv(general_configuration["general_parameters"]["train_dataset_path"])
    val_dataframe = pd.read_csv(
        general_configuration["general_parameters"]["validation_dataset_path"]
    )

    # TODO: add instanciation of probing model when ready
    # if probing load pre-computed embeddings else load (DNA, RNA or Protein) sequences
    if general_configuration["model_architecture"]["model_type"] == "probing":
        pass

    # if finetuning, or peft, load sequences from csv file
    else:
        # initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_hf_model_path)

        model_configuration = {
            "shuffle_dataloader": general_configuration["training_parameters"][
                "shuffle_dataloader"
            ],
            "k_for_kmers": general_configuration["model_architecture"]["k_for_kmers"],
            "llm_hf_model_path": llm_hf_model_path,
            "mlp_layer_sizes": None,
            "embed_dim": 1024,
            "lr": general_configuration["training_parameters"]["lr"],
            "batch_size": general_configuration["training_parameters"]["batch_size"],
            "num_epochs": general_configuration["training_parameters"]["num_epochs"],
        }
        model = LLMBasedModel(
            parameters=model_configuration,
            folder_name=folder_name,
            model_type=model_type,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            checkpoints=Path(""),
        )

    model.fit(train_dataframe, val_dataframe)
