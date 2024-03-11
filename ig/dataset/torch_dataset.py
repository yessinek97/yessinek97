"""Module used to define the torch dataset."""
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
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


class PeptidePairsDataset(torch.utils.data.Dataset):
    """Pytorch dataset class for finetuning.

    Args:
        torch.utils.data.Datasets (_type_): _description_.
    """

    def __init__(
        self,
        mutated_peptides: list,
        wild_type_peptides: list,
        labels: list,
        tokenizer: AutoTokenizer,
        max_length: int = 2001,
    ):
        """Initializes the dataset object.

        Args:
            mutated_peptides (list[str]): list of sequences of mutated peptides.
            wild_type_peptides (list[str]): list of sequences of wild_type peptides.
            tokenizer (AutoTokenizer): used to tokenize sequences
            labels (list[int]): IG labels of the (wild_type, mutated) sequence pairs.
            max_length (int): maximum treatable length.
        """
        self._mutated_peptides = mutated_peptides
        self._wild_type_peptides = wild_type_peptides
        self._labels = labels
        self._max_length = max_length
        self._tokenizer = tokenizer

    # used as collate_fn when creating the dataloader
    def tokenize_batch_of_pairs(self, pairs_batch: list) -> Tuple[torch.Tensor, np.ndarray]:
        """Tokenzies a batch of pairs of (wild_type, mutated) sequences.

        Args:
            pairs_batch (list): batch of (wild_type, mutated) sequence pairs

        Returns:
            torch.Tensor: tokenized batch of (wild_type, mutated) sequence pairs.
            np.ndarray: array of labels
        """
        batch_sequence_pairs, batch_labels = zip(*pairs_batch)

        # flatten batch sequence pairs to be tokenized in one pass
        flattened_batch_sequence_pairs = [seq for pair in batch_sequence_pairs for seq in pair]

        batch_pair_token_ids = self._tokenizer.batch_encode_plus(
            flattened_batch_sequence_pairs,
            return_tensors="pt",
            padding="max_length",
            max_length=self._max_length,
        )["input_ids"]

        # reshape batch token_ids back in pair-wise shape
        batch_pair_token_ids = torch.reshape(batch_pair_token_ids, (-1, 2, self._max_length))

        return batch_pair_token_ids, torch.Tensor(batch_labels)

    def __len__(self) -> int:
        """Returns the number of samples in the whole dataset."""
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], list]:
        """Returns a pair of (wild_type, mutated) and corresponding label.

        Args:
            idx (int): index or list of indexes of pairs to return.

        Returns:
            Tuple[Tuple[str, str], list]: tokenized sequenes pair and label.
        """
        pair = (self._wild_type_peptides[idx], self._mutated_peptides[idx])
        label = self._labels[idx]
        return pair, label
