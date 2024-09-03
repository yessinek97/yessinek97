"""Trainer used to train llm based models."""
import os
from pathlib import Path

import click
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ig import CONFIGURATION_DIRECTORY
from ig.dataset.torch_dataset import PeptidePairsDataset
from ig.models.torch_based_models import FinetuningModel
from ig.utils.io import load_yml
from ig.utils.logger import get_logger
from ig.utils.torch import TOKENIZER_SOURCES

log = get_logger("dl_model/inference")


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
    model_source = general_configuration["model_config"]["general_params"]["model_source"]
    tokenizer_source = general_configuration["model_config"]["general_params"]["tokenizer_source"]
    training_type = general_configuration["model_config"]["general_params"]["training_type"]
    model_configuration = general_configuration["model_config"]["model_params"]
    wildtype_col_name = general_configuration["model_config"]["general_params"]["wildtype_col_name"]
    mutated_col_name = general_configuration["model_config"]["general_params"]["mutated_col_name"]
    mutation_position_col_name = general_configuration["model_config"]["general_params"][
        "mutation_position_col_name"
    ]
    context_length = general_configuration["model_config"]["model_params"]["context_length"]
    batch_size = general_configuration["model_config"]["model_params"]["batch_size"]
    shuffle_dataloader = general_configuration["model_config"]["model_params"]["shuffle_dataloader"]

    # read data
    dataframe = pd.read_csv(dataset_path)
    wild_type_sequences = list(dataframe[wildtype_col_name])
    mutated_sequences = list(dataframe[mutated_col_name])

    labels = list(dataframe["cd8_any"])

    # crop sequences to match desired context length
    if context_length:
        if mutation_position_col_name and mutation_position_col_name in list(dataframe.columns):
            mutation_start_positions = list(dataframe[mutation_position_col_name])
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
        else:
            raise ValueError(
                "if context length is set, mutation positions should be provided"
                "and the corresponding column should exist"
            )

    # number of tokens of the longest sequence
    tokenizer_source = TOKENIZER_SOURCES[tokenizer_source]
    tokenizer = tokenizer_source.from_pretrained(llm_hf_model_path)
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
        batch_size=batch_size,
        shuffle=shuffle_dataloader,
        collate_fn=peptide_pairs_dataset.tokenize_batch_of_pairs,
    )

    model = FinetuningModel(
        llm_hf_model_path=llm_hf_model_path,
        model_source=model_source,
        model_configuration=model_configuration,
        training_type=training_type,
        tokenizer=tokenizer,
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # selecting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    if use_cuda & (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    log.info(f"starting inference using: {device}")
    with torch.no_grad():
        all_probs = []
        for pair, _ in tqdm(peptide_pairs_dataloader):
            pair = pair.to(device)
            logits = model(pair=pair, max_length=max_length, attention_mask=None)
            probs = torch.nn.Sigmoid()(logits)
            all_probs += list(probs.detach().cpu().numpy())

    if save_preds:
        dataframe["llm_based_prediction"] = all_probs
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        df_name = dataset_path.split("/")[-1]
        target_path = Path(os.path.join(folder_name, df_name))
        log.info(f"Concatenating probs and saving new dataframe to: {target_path}")
        dataframe.to_csv(target_path)
