"""Module used to define the torch dataset."""
import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class OneSequenceDatasetForEmbedding(Dataset):
    """Pytorch dataset class for embedding computation.

    Args:
        torch.utils.data.Datasets (_type_): _description_.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_column: str,
        tokenizer: AutoTokenizer,
        cf_processing: Dict[str, Any],
    ) -> None:
        """Initializes the object with the provided DataFrame, sequences, index, and tokenizer.

        Args:
            df (pd.DataFrame): The DataFrame containing the sequences.
            seq_column (str): The column name of the sequences in the DataFrame.
            tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
            cf_processing (Dict[str, Any]): The configuration for sequence processing.
        """
        self.tokenizer = tokenizer
        self.cf_processing = cf_processing

        df = df[df[seq_column].notna()]
        self.max_len = df[seq_column].apply(len).max()

        if self.cf_processing["drop_duplicates"]:
            df = df.drop_duplicates(seq_column)

        self.id = df[seq_column].tolist()
        if self.cf_processing["separate_tokens"]:
            df[seq_column] = df[seq_column].apply(self.separate_tokens)
        if self.cf_processing["replace"]:
            df[seq_column] = df[seq_column].apply(self.process_seq)

        self.sequences = df[seq_column].tolist()

    def __len__(self) -> int:
        """Return the length of input sequences."""
        return len(self.sequences)

    def __getitem__(self, i: int) -> Tuple[str, int]:
        """Get an item by index.

        Args:
            i (int): The index of the item to retrieve.

        Returns:
            Tuple[str, int]: A tuple containing the sequence and its associated ID.
        """
        return self.sequences[i], self.id[i]

    def batch_tokenizer(
        self, batch: List[Tuple[str, int]]
    ) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """Batch tokenizer for the dataset.

        Args:
            batch (List[Tuple[str,int]]): The batch to be tokenized.

        Returns:
            Dict[str, Union[torch.Tensor, List[int]]]: The tokenized batch.
        """
        sequences, ids = zip(*batch)
        return {
            "encoded_inputs": self.tokenizer(
                sequences,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_len,
                truncation=True,
            ),
            "id": ids,
        }

    def process_seq(self, x: str) -> str:
        """Replace the occurrences of 'U', 'Z', 'O', and 'B' with 'X' in the input string x.

        Args:
            x (str): The input string to process.

        Returns:
            str: The processed string with replacements.
        """
        return re.sub(self.cf_processing["replace_pattern"], self.cf_processing["replace_with"], x)

    def separate_tokens(self, x: str) -> str:
        """Join the elements of the input list with a space.

        Args:
        x (list): The list of tokens to be joined.

        Returns:
        str: The joined string with elements separated by a space.
        """
        return " ".join(x)
