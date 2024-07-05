"""Module used to define some helper functions for LLM models."""
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch

from ig import DATA_DIRECTORY, DATAPROC_DIRECTORY, EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY
from ig.utils.general import log
from ig.utils.io import load_pkl, load_yml, save_yml


# Loads an embeddings pickle file for probing experiment
def load_embedding_file(file_path: str) -> Any:
    """Load pre generated sequence embeddings from a pickle file.

    Args:
        file_path (str): Path to the embedding file

    Raises:
        FileNotFoundError: raise an error if the file can't be found

    Returns:
        Any: return an object with the file
    """
    local_path = DATA_DIRECTORY / file_path
    if Path(local_path).exists():
        log.info("Loading embedding pickle file at: %s \n", local_path)
        return load_pkl(local_path)
    raise FileNotFoundError("The embeddings pickle file %s cannot be found!!!\n" % local_path)


def copy_embeddings_file(source_path: Path, destination_path: Path) -> bool:
    """Save embeddings file when doing probing to the current experiment directory.

    Args:
        source_path (Path): source embedding file path
        destination_path (Path): destination path inside the current data_proc directory

    Returns:
        bool: retrun true if the file was copied correctly
    """
    # shutil.copy2() method copies the source file to the destination directory with it's metadata
    dst = shutil.copy2(source_path, destination_path)

    return dst.exists()


def change_embeddings_path(config_path: Path, new_path: Path) -> bool:
    """Change emb_file_path in llm models config file to the new file that was copyed inside the experiment's data_proc directory.

    Args:
        config_path (Path): llm models config file path
        new_path (Path): the path of the copied embeddings file

    Returns:
        bool: return true if the change was successfull
    """
    config = load_yml(config_path)

    config["model_config"]["general_params"]["emb_file_path"] = str(new_path)

    save_yml(config, config_path)

    return config_path.exists()


def save_probing_embeddings(experiment_path: Path, emb_file_path: Path) -> None:
    """Check if the current experiment has an LLM probing model then save it's embeddings file to the models directory.

    Args:
        experiment_path (Path): Path to the current experiment
        emb_file_path (Path): Path to the probing embeddings file

    Returns:
        None
    """
    emb_file_path = DATA_DIRECTORY / emb_file_path
    file_name = Path(emb_file_path).name
    emb_dest_path = experiment_path / DATAPROC_DIRECTORY / file_name

    # Copy the embeddings file to data_proc/ inside the current experiment directory
    saved = copy_embeddings_file(emb_file_path, emb_dest_path)

    # Change the embeddings path in current experiment llm model config
    llm_model_conf_path = (
        experiment_path / EXPERIMENT_MODEL_CONFIGURATION_DIRECTORY / "llm_based_probing_config.yml"
    )
    changed = change_embeddings_path(llm_model_conf_path, emb_dest_path)

    if saved & changed:
        log.info("Embeddings file was successfully copied to : %s", emb_dest_path)
    else:
        log.warning("The embeddings file for probing experiment was not saved!!!")


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
