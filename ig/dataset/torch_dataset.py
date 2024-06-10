"""Module used to define the torch dataset."""
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ig.utils.logger import get_logger

log = get_logger("Train/model/torch_dataset")


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
        wild_type_peptides: list,
        mutated_peptides: list,
        labels: list,
        tokenizer: AutoTokenizer,
        max_length: int = 2001,
    ):
        """Initializes the dataset object.

        Args:
            wild_type_peptides (list[str]): list of sequences of wild_type peptides.
            mutated_peptides (list[str]): list of sequences of mutated peptides.
            tokenizer (AutoTokenizer): used to tokenize sequences
            labels (list[int]): IG labels of the (wild_type, mutated) sequence pairs.
            max_length (int): maximum treatable length.
        """
        self._wild_type_peptides = wild_type_peptides
        self._mutated_peptides = mutated_peptides
        self._max_length = max_length
        self._tokenizer = tokenizer
        if labels == []:
            self._labels = [float("inf") for _ in range(len(mutated_peptides))]
        else:
            self._labels = labels

    # used as collate_fn when creating the dataloader
    def tokenize_batch_of_pairs(self, pairs_batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
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

        batch_pair_token_ids = torch.Tensor(
            self._tokenizer(
                flattened_batch_sequence_pairs,
                padding="max_length",
                max_length=self._max_length,
            )["input_ids"]
        ).to(torch.int64)

        # reshape batch token_ids back in pair-wise shape
        batch_pair_token_ids = torch.reshape(batch_pair_token_ids, (-1, 2, self._max_length))

        return batch_pair_token_ids, torch.Tensor(batch_labels)

    def __len__(self) -> int:
        """Returns the number of samples in the whole dataset."""
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], float]:
        """Returns a pair of (wild_type, mutated) and corresponding label.

        Args:
            idx (int): index or list of indexes of pairs to return.

        Returns:
            Tuple[Tuple[str, str], list]: tokenized sequenes pair and label.
        """
        pair = (self._wild_type_peptides[idx], self._mutated_peptides[idx])
        label = self._labels[idx]
        return pair, label


class MixedDataset(torch.utils.data.Dataset):
    """Pytorch dataset class for mixed approach.

    Args:
        torch.utils.data.Datasets (_type_): _description_.
    """

    def __init__(
        self,
        wild_type_peptides: list,
        mutated_peptides: list,
        tabular_features: Dict[str, list],
        labels: list,
        llm: AutoModelForMaskedLM,
        tokenizer: AutoTokenizer,
        device: str,
        max_length: int = 2001,
    ):
        """Initializes the dataset object.

        Args:
            wild_type_peptides (list[str]): list of sequences of wild_type peptides.
            mutated_peptides (list[str]): list of sequences of mutated peptides.
            tabular_features (dict[str, list]): dictionnary containing tabular featuer values
            labels (list[int]): IG labels of the (wild_type, mutated) sequence pairs.
            llm (AutoModelFroMaskedLM): LLM to compute sequence embeddigs.
            tokenizer (AutoTokenizer): used to tokenize sequences
            device (str): device on which the embeddings will be computed
            max_length (int): maximum treatable length.
        """
        self._wild_type_peptides = wild_type_peptides
        self._mutated_peptides = mutated_peptides
        self._tabular_features = tabular_features
        self._llm = llm
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        if labels == []:
            self._labels = [float("inf") for _ in range(len(mutated_peptides))]
        else:
            self._labels = labels

    def compute_batch_mutation_embeddings(
        self, pairs_batch: list, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes embeddings for a batch of pairs of (wild_type, mutated) sequences.

        Args:
            pairs_batch (list): batch of (wild_type, mutated) sequence pairs

        Returns:
            torch.Tensor: batch embeddings of (wild_type, mutated) sequence pairs.
            torch.Tensor: tensor of batch features
            torch.Tensor: tensor of batch labels
        """
        batch_sequence_pairs, batch_features, batch_labels = zip(*pairs_batch)
        batch_size = len(batch_sequence_pairs)

        # flatten batch sequence pairs to be tokenized in one pass
        flattened_batch_sequence_pairs = [seq for pair in batch_sequence_pairs for seq in pair]

        # batch tokenized pairs
        x = torch.Tensor(
            self._tokenizer(
                flattened_batch_sequence_pairs,
                padding="max_length",
                max_length=self._max_length,
            )["input_ids"]
        ).to(torch.int64)

        if not attention_mask:
            attention_mask = x != self._tokenizer.cls_token_id

        # batch embedding pairs
        x = self._llm(
            x.view(batch_size * 2, -1),  # flatten along batch to compute all embeddings in one pass
            attention_mask=attention_mask.view(batch_size * 2, -1),
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )["hidden_states"][-1]

        # group embeddings along batch
        x = torch.reshape(x, (batch_size, 2, self._max_length, -1))

        # mutation embedding given by the difference between the mutated and wild type
        mutation_embeddings = x[:, 0, :, :] - x[:, 1, :, :]

        # calculate attention mask of the mutation embeddings in a way that positions
        # that are masked on either of the wild type or mutated sequences,
        # is masked on the final (mutation) sequence
        mutation_mask = torch.unsqueeze(
            torch.mul(attention_mask[0, :], attention_mask[1, :]),
            dim=-1,
        ).to(self._device)

        aggregated_embedding = torch.sum(mutation_mask * mutation_embeddings, axis=-2) / torch.sum(
            mutation_mask
        )

        return aggregated_embedding, batch_features, batch_labels

    def __len__(self) -> int:
        """Returns the number of samples in the whole dataset."""
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, str], list, float]:
        """Returns a pair of (wild_type, mutated) and corresponding label.

        Args:
            idx (int): index or list of indexes of pairs to return.

        Returns:
            Tuple[Tuple[str, str], list, list]: tokenized sequenes pair, tabular features and label.
        """
        sequence_pair = (self._wild_type_peptides[idx], self._mutated_peptides[idx])
        features = [
            self._tabular_features[feature][idx] for feature in self._tabular_features.keys()
        ]
        label = self._labels[idx]
        return sequence_pair, features, label


class EmbeddingsPairsDataset(torch.utils.data.Dataset):
    """Pytorch dataset class for loading embedded sequences.

    Args:
        torch.utils.data.Datasets (_type_): .
    """

    def __init__(
        self,
        wild_type_sequences: list,
        mutated_sequences: list,
        embeddings: Dict,
        labels: list,
    ):
        """Initializes the embeddings dataset object.

        Args:
            wild_type_sequences (list): list of wild type sequences
            mutated_sequences (list): list of mutated sequences
            embeddings (Dict): dictionary containing sequences as keys and their embeddings
            labels (list): list of labels (floats)
            config (Dict[str, Any]): dictionary containing the configuration parameters
        """
        self._wt_seq = wild_type_sequences
        self._mt_seq = mutated_sequences
        self._embeddings = embeddings
        if labels == []:
            self._labels = [float("inf") for _ in range(len(mutated_sequences))]
        else:
            self._labels = labels

    def __len__(self) -> int:
        """Returns the number of samples in the whole dataset."""
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """Returns a pair of embeddings (wt_emb, mt_emb) and corresponding label.

        Args:
            idx (int): index of pairs to return.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: (wt_seq, mt_seq) and label.
        """
        # use the sample ID to get it's wt_seq, mt_seq
        wt_seq = self._wt_seq[idx]
        mt_seq = self._mt_seq[idx]
        # Use the seq as key to get it's emmbedding vector
        wt_emb = self._embeddings[wt_seq]
        mt_emb = self._embeddings[mt_seq]
        label = self._labels[idx]
        embeddings_pair = (wt_emb, mt_emb)
        embeddings_pair = torch.Tensor(embeddings_pair)
        label = float(label)

        return embeddings_pair, label
