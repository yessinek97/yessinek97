"""Trainer used to train llm based models."""
import os
from pathlib import Path

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ig import CONFIGURATION_DIRECTORY
from ig.dataset.torch_dataset import PeptidePairsDataset
from ig.src.llm_based_models import FinetuningModel
from ig.src.utils import load_yml


@click.command()
@click.option(
    "--configuration-file",
    "-c",
    type=str,
    required=True,
    help=" Path to configuration file.",
)
@click.option("--dataset-path", "-data", type=str, required=True, help="Path to dataset file")
@click.option(
    "--checkpoint-path", "-ckpt", type=str, required=True, help="Path to model checkpoint"
)
@click.option(
    "--save-preds",
    type=bool,
    required=True,
    help="whether or not to concat predictions to dataframe",
)
@click.option(
    "--folder-name",
    "-n",
    type=str,
    required=True,
    help="path to folder where the experiment output should be saved",
)
def predict(
    folder_name: str,
    configuration_file: str,
    dataset_path: str,
    checkpoint_path: str,
    save_preds: str,
) -> None:
    """Mehtod to compute IG predictions using the chosen model.

    Args:
        folder_name (str): Path to folder where the experiment output will be saved.
        configuration_file (str): Name of the config file to be used.
        dataset_path (str): Path to dataset.
        checkpoint_path (str): Path to model checkpoint.
        save_preds (str): whether or not to concat predictions to dataframe
    """
    # read configuration
    general_configuration = load_yml(CONFIGURATION_DIRECTORY / "model_config" / configuration_file)
    llm_hf_model_path = general_configuration["model_config"]["general_params"]["llm_hf_model_path"]
    is_masked_model = general_configuration["model_config"]["general_params"]["is_masked_model"]
    training_type = general_configuration["model_config"]["general_params"]["training_type"]
    model_configuration = general_configuration["model_config"]["model_params"]
    tokenizer = AutoTokenizer.from_pretrained(llm_hf_model_path)

    # read data
    dataframe = pd.read_csv(dataset_path)
    wild_type_sequences = list(dataframe["wild_type"])
    mutated_sequences = list(dataframe["mutated"])
    mutation_start_positions = list(dataframe["mutation_start_position"])
    labels = list(dataframe["cd8_any"])

    # crop sequences to match desired context length
    if general_configuration["model_config"]["model_params"]["context_length"]:
        context_length = general_configuration["model_config"]["model_params"]["context_length"]

        wild_type_sequences = [
            seq[
                max(0, mut_pos - context_length // 2) : max(0, mut_pos - context_length // 2)
                + context_length
            ]
            for seq, mut_pos in zip(wild_type_sequences, mutation_start_positions)
        ]

        mutated_sequences = [
            seq[
                max(0, mut_pos - context_length // 2) : max(0, mut_pos - context_length // 2)
                + context_length
            ]
            for seq, mut_pos in zip(mutated_sequences, mutation_start_positions)
        ]

    # number of tokens of the longest sequence
    max_length = max(
        [
            len(tokenizer.encode_plus(seq)["input_ids"])
            for seq in tqdm(wild_type_sequences + mutated_sequences)
        ]
    )

    # instanciate PeptidePairs dataset
    peptide_pairs_dataset = PeptidePairsDataset(
        wild_type_sequences, mutated_sequences, labels, tokenizer, max_length
    )
    # instanciate PeptidePairs dataloader
    peptide_pairs_dataloader = DataLoader(
        peptide_pairs_dataset,
        batch_size=general_configuration["model_config"]["model_params"]["batch_size"],
        shuffle=general_configuration["model_config"]["model_params"]["shuffle_dataloader"],
        collate_fn=peptide_pairs_dataset.tokenize_batch_of_pairs,
    )

    model = FinetuningModel(
        llm_hf_model_path=llm_hf_model_path,
        is_masked_model=is_masked_model,
        model_configuration=model_configuration,
        training_type=training_type,
        tokenizer=tokenizer,
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # selectkng device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"starting inference using: {device}")
    with torch.no_grad():
        all_probs = []
        for pair, _ in tqdm(peptide_pairs_dataloader):
            pair = pair.to(device)
            logits = model(pair, max_length)
            probs = torch.nn.Sigmoid()(logits)
            all_probs += list(probs.detach().cpu().numpy())

    if save_preds:
        dataframe["llm_based_prediction"] = all_probs
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        df_name = dataset_path.split("/")[-1]
        target_path = Path(os.path.join(folder_name, df_name))
        print(f"Concatenating probs and saving new dataframe to: {target_path}")
        dataframe.to_csv(target_path)
