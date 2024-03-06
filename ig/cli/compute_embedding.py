"""Module used to compute embedding for a given datasets."""
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

import click
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from ig import CONFIGURATION_DIRECTORY
from ig.dataset.torch_dataset import OneSequenceDatasetForEmbedding
from ig.src.logger import get_logger, init_logger
from ig.src.utils import load_yml, read_data, save_as_pkl
from ig.utils.torch_helper import get_device

log: Logger = get_logger("ComputeEmbedding")


@click.command()
@click.option(
    "--input_file",
    "-f",
    type=str,
    required=True,
    help="Path to the dataset",
)
@click.option(
    "--configuration_file", "-c", type=str, required=True, help=" Path to configuration file."
)
def compute_embedding(input_file: str, configuration_file: str) -> None:
    """Compute embedding for given input file and configuration file.

    Args:
        input_file (str): Path to the input dataset.
        configuration_file (str): Path to the configuration file.

    Returns:
        None
    """
    configuration = load_yml(CONFIGURATION_DIRECTORY / configuration_file)
    config_seq_proc = configuration["processing"]
    model_name = configuration["model"]
    is_masked_model = configuration.get("is_masked_model", False)
    data_path = Path(input_file)
    data_name = data_path.stem
    data_directory = data_path.parent / "embedding" / model_name
    data_directory.mkdir(parents=True, exist_ok=True)
    # read data
    data = read_data(input_file)
    init_logger(logging_directory=data_directory)

    seq_columns = configuration["seq_columns"]
    output_types = configuration["outputs"]
    check_output_type(output_types)
    batch_size = configuration["batch_size"]
    workers = configuration["workers"]
    seq_columns = seq_columns if isinstance(seq_columns, list) else [seq_columns]
    output_types = output_types if isinstance(output_types, list) else [output_types]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=False, trust_remote_code=True
    )

    if is_masked_model:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, output_hidden_states=True
        )
    else:
        model = AutoModel.from_pretrained(model_name)
    log.info("Start computing embedding for %s", model_name)
    log.info("Output format %s", ",".join(output_types))
    # set model to eval mode
    model.eval()
    # set model device
    device = get_device()
    model.to(device)
    for seq_column in seq_columns:
        log.info("Start computing embedding for %s", seq_column)
        dataset = OneSequenceDatasetForEmbedding(
            df=data,
            seq_column=seq_column,
            tokenizer=tokenizer,
            cf_processing=config_seq_proc,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            drop_last=False,
            collate_fn=dataset.batch_tokenizer,
        )
        ids = []
        output: List[Dict[str, torch.Tensor]] = []
        for batch in tqdm(dataloader):
            encoded_inputs = {
                k: batch["encoded_inputs"][k].to(device) for k in batch["encoded_inputs"].keys()
            }
            batch_ids = batch["id"]
            with torch.no_grad():
                batch_embedding = model(**encoded_inputs)
                attention_mask = (encoded_inputs["input_ids"] != tokenizer.cls_token_id) & (
                    encoded_inputs["input_ids"] != tokenizer.pad_token_id
                )
                batch_output = process_batch_embedding(
                    batch_embedding, attention_mask, output_types, is_masked_model
                )
                output.append(batch_output)
                ids.extend(batch_ids)

        for output_type in output_types:
            log.info("Start saving embedding for %s type", output_type)
            embedding_output = [emb[output_type] for emb in output]
            embedding_output = torch.concat(embedding_output, axis=0).cpu().numpy()

            output_per_type = {}
            for seq_id, emb in zip(ids, embedding_output):
                output_per_type[seq_id] = emb
            save_as_pkl(
                output_per_type, data_directory / f"{data_name}_{seq_column}_{output_type}.pkl"
            )
        log.info("End computing embedding for %s", seq_column)

    log.info("Files saved in %s", data_directory)


def process_batch_embedding(
    batch_embedding: Any,
    attention_mask: torch.Tensor,
    output_types: List[str],
    is_masked_model: bool,
) -> Dict[str, torch.Tensor]:
    """Process batch embedding and return a dictionary of output types.

    Args:
        batch_embedding (Any): The batch embedding to process.
        attention_mask (torch.Tensor): The attention mask to apply.
        output_types (List[str]): The types of output to include in the returned dictionary.

    Returns:
        Dict[str, torch.Tensor]: The processed batch embedding.
    """
    batch_output: Dict[str, torch.Tensor] = {}

    if is_masked_model:
        last_hidden_state = batch_embedding["hidden_states"][-1]
    else:
        last_hidden_state = batch_embedding.last_hidden_state
    if "mean" in output_types:
        masked_outputs = last_hidden_state * attention_mask.unsqueeze(-1)
        emb = torch.sum(masked_outputs, dim=1) / torch.sum(attention_mask, axis=1).unsqueeze(-1)
        batch_output["mean"] = emb
    if "cls" in output_types:
        batch_output["cls"] = last_hidden_state[:, 0, :]
    if "raw" in output_types:
        batch_output["raw"] = last_hidden_state

    return batch_output


def check_output_type(output_type: str) -> None:
    """Function to check the output type and raise a ValueError.

    Raises a ValueError if it is not 'mean', 'cls', or 'raw'.

    Args:
    output_type (str): The type of output to be checked.
    """
    for o_type in output_type:
        if o_type not in ["mean", "cls", "raw"]:
            raise ValueError(f"output type {o_type} not in ['mean','cls','raw']")
