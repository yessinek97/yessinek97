"""File that contains the LLM model that will be used in finetuning, probing or peft experiments."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ig.dataset.torch_dataset import EmbeddingsPairsDataset, PeptidePairsDataset
from ig.models.base_model import BaseModel, log
from ig.models.torch_based_models import FinetuningModel, ProbingModel
from ig.src.torch_utils import set_torch_reproducibility
from ig.src.utils import crop_sequences, load_embedding_file, save_as_pkl


class LLMModel(BaseModel):
    """Classifies the aggregated embeddings of the mutated and wild type sequences."""

    def __init__(
        self,
        parameters: dict[str, Any],
        folder_name: str,
        model_type: str,
        experiment_name: str,
        dataset_name: str,
        checkpoints: Path,
        features: list[str],
        label_name: str,
        prediction_name: str,
        other_params: dict[str, Any],
        attention_mask: torch.Tensor | None = None,
        save_model: bool = False,
    ):
        """Initializes LLMModel.

        Args:
            parameters (dict[str, Any]): _description_
            folder_name (str): _description_
            model_type (str): _description_
            experiment_name (str): _description_
            dataset_name (str): _description_
            tokenizer (AutoTokenizer): _description_
            checkpoints (Path): _description_
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            save_model (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_
        """
        super().__init__(
            parameters=parameters,
            features=features,
            label_name=label_name,
            prediction_name=prediction_name,
            other_params=other_params,
            folder_name=folder_name,
            model_type=model_type,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            checkpoints=checkpoints,
            save_model=save_model,
        )

        self._attention_mask = attention_mask

        self._training_type = other_params["training_type"]
        self.check_training_type()
        log.info("Training type: %s", self._training_type)
        # set torch in deterministic mode (for reproducibility)
        set_torch_reproducibility(self.parameters["seed"])

        if self._training_type in ["finetuning", "peft"]:
            self._tokenizer = AutoTokenizer.from_pretrained(other_params["llm_hf_model_path"])
            self._llm_based_model = FinetuningModel(
                model_configuration=parameters,
                llm_hf_model_path=other_params["llm_hf_model_path"],
                is_masked_model=other_params["is_masked_model"],
                tokenizer=self._tokenizer,
                training_type=self._training_type,
            )
        elif self._training_type == "probing":
            self._llm_based_model = ProbingModel(model_configuration=parameters)
            # Load embeddings from pickle file
            self._embeddings_pkl = load_embedding_file(other_params["emb_file_path"])

        # only save finetuned backbone to use for another task
        # instead of saving the whole LLMModel
        self._save_llm_only = other_params["save_llm_only"]

        self._use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        if self._use_cuda:
            # send model to GPU and enable multi-gpu usage
            self._llm_based_model = torch.nn.DataParallel(self._llm_based_model).to(self._device)

        self._wildtype_col_name = other_params["wildtype_col_name"]
        self._mutated_col_name = other_params["mutated_col_name"]
        self._mutation_position_col_name = other_params["mutation_position_col_name"]

        self._prob_threshold = parameters["threshold"]
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer = Adam(self._llm_based_model.parameters(), lr=self.parameters["lr"])
        self._metrics: dict[str, list[float]] = defaultdict(list)

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
    ) -> None:
        """Fit method."""
        num_trainable_params = sum(
            p.numel() for p in self._llm_based_model.parameters() if p.requires_grad
        )
        log.info(
            f"Started training: {self.model_type} | using: {self._device}\n\
            number of trainable parameter: {num_trainable_params}"
        )

        train_dataloader, max_length_train = self._create_matrix(train_data, with_label=True)
        val_dataloader, max_length_val = self._create_matrix(val_data, with_label=True)

        # number of tokens of the longest sequence
        log.info(
            f"Maximum length in train: {max_length_train} - "
            f"Maximum length in validation: {max_length_val}"
        )

        num_epochs = self.parameters["num_epochs"]
        epochs_iterator = tqdm(range(num_epochs))
        for epoch_num in epochs_iterator:
            epochs_iterator.set_description(f"epoch: {epoch_num+1}/{num_epochs}")
            self.train_one_epoch(train_dataloader, max_length_train)
            self.validate_one_epoch(val_dataloader, max_length_val)
            epochs_iterator.set_postfix({"Train accuracy": self._metrics["train_acc"][epoch_num]})
            self.log_metrics(epoch_num)

        if self.save_model:
            log.info(f"Saving model to: {self.checkpoints}")
            # Save torch model checkpoint
            if self._save_llm_only:
                dict_to_save = {"model_state_dict": self._llm_based_model.module.llm.state_dict()}
            else:
                dict_to_save = {
                    "model_state_dict": self._llm_based_model.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "metrics": self._metrics,
                }
            torch.save(dict_to_save, str(self.checkpoints).replace("model.pkl", "torch_model.pt"))
            # pickle LLMModel object
            save_as_pkl(self, self.checkpoints)

    def train_one_epoch(self, train_dataloader: DataLoader, max_length: int) -> None:
        """Train model for one epoch.

        Args:
            train_dataloader (DataLoader): _description_
            max_length (int): _description_
        """
        self._llm_based_model.train()
        epoch_preds = []
        epoch_labels = []
        iterator = tqdm(train_dataloader)
        for pair, label in iterator:
            batch_loss, batch_preds, batch_labels = self.train_one_step(pair, label, max_length)
            iterator.set_postfix({"Loss": batch_loss.item()})
            epoch_preds += batch_preds
            epoch_labels += batch_labels

        epoch_preds = torch.Tensor(epoch_preds).detach().cpu()
        epoch_labels = torch.Tensor(epoch_labels).detach().cpu()

        positive_indexes = [idx for idx in range(len(epoch_labels)) if epoch_labels[idx] == 1]
        epoch_pos_preds = [
            1 if epoch_preds[idx] == epoch_labels[idx] else 0 for idx in positive_indexes
        ]

        self._metrics["train_acc"].append(np.average(epoch_preds == epoch_labels))
        self._metrics["train_acc_pos"].append(np.average(epoch_pos_preds))

    def train_one_step(
        self, pair: torch.Tensor, label: torch.Tensor, max_length: int
    ) -> tuple[torch.Tensor, list, list]:
        """Make one forward pass and backpropagates the loss of one batch.

        Args:
            pair (torch.Tensor): _description_
            label (torch.Tensor): _description_
            max_length (int): _description_
        """
        pair = pair.to(self._device)
        label = label.to(self._device)

        # forward pass
        logits = self._llm_based_model(
            pair=pair, max_length=max_length, attention_mask=self._attention_mask
        )

        probs = torch.nn.Sigmoid()(logits)
        prediction = (
            probs.detach()
            .cpu()
            .apply_(
                lambda prob: math.floor(prob) if prob < self._prob_threshold else math.ceil(prob)
            )
            .squeeze(-1)
        )

        # calculate step loss
        batch_loss = self._criterion(logits.squeeze(1), label)

        self._metrics["train_loss_per_step"].append(batch_loss.item())

        # back propagate loss and update gradient
        batch_loss.backward()
        self._optimizer.step()

        return batch_loss, prediction, label

    def validate_one_epoch(self, val_dataloader: DataLoader, max_length: int) -> None:
        """Validate model for one epoch.

        Args:
            val_dataloader (DataLoader): _description_
            max_length (int): _description_


        Returns:
            _type_: _description_
        """
        self._llm_based_model.eval()
        with torch.no_grad():
            epoch_preds = []
            epoch_labels = []
            iterator = tqdm(val_dataloader)
            for pair, label in iterator:
                batch_loss, batch_preds, batch_labels = self.validate_one_step(
                    pair, label, max_length
                )
                iterator.set_postfix({"Loss": batch_loss.item()})
                epoch_preds += batch_preds
                epoch_labels += batch_labels

        epoch_preds = torch.Tensor(epoch_preds).detach().cpu()
        epoch_labels = torch.Tensor(epoch_labels).detach().cpu()

        positive_indexes = [idx for idx in range(len(epoch_labels)) if epoch_labels[idx] == 1]
        epoch_pos_preds = [
            1 if epoch_preds[idx] == epoch_labels[idx] else 0 for idx in positive_indexes
        ]

        self._metrics["val_acc"].append(np.average(epoch_preds == epoch_labels))
        self._metrics["val_acc_pos"].append(np.average(epoch_pos_preds))

    def validate_one_step(
        self, pair: torch.Tensor, label: torch.Tensor, max_length: int
    ) -> tuple[torch.Tensor, list, list]:
        """Make one forward pass.

        Args:
            pair (torch.Tensor): _description_
            label (torch.Tensor): _description_
            max_length (int): _description_
        """
        pair = pair.to(self._device)
        label = label.to(self._device)

        # forward pass
        logits = self._llm_based_model(
            pair=pair, max_length=max_length, attention_mask=self._attention_mask
        )

        probs = torch.nn.Sigmoid()(logits)
        prediction = (
            probs.detach()
            .cpu()
            .apply_(
                lambda prob: math.floor(prob) if prob < self._prob_threshold else math.ceil(prob)
            )
            .squeeze(-1)
        )

        # calculate step loss
        batch_loss = self._criterion(logits.squeeze(1), label)
        self._metrics["val_loss_per_step"].append(batch_loss.item())

        return batch_loss, prediction, label

    def log_metrics(self, epoch_num: int) -> None:
        """Log epoch metrics.

        Args:
            epoch_num (_type_): _description_
        """
        epoch_train_acc = self._metrics["train_acc"][epoch_num]
        epoch_val_acc = self._metrics["val_acc"][epoch_num]
        epoch_train_acc_pos = self._metrics["train_acc_pos"][epoch_num]
        epoch_val_acc_pos = self._metrics["val_acc_pos"][epoch_num]

        log.info(
            f"Epoch {epoch_num + 1} metrics: \
            | Train Acc: {epoch_train_acc: .3f} | Train Acc Positives: {epoch_train_acc_pos: .3f} \
            | Val Acc: {epoch_val_acc: .3f} | Val Acc Positives: {epoch_val_acc_pos: .3f}"
        )

    def predict(self, data: pd.DataFame, with_label: bool) -> Any:
        """Prediction method."""
        inference_dataloader, max_length = self._create_matrix(data, with_label)
        log.info(f"Started inference using: {self._device}")

        predictions = []
        with torch.no_grad():
            for pair, _ in tqdm(inference_dataloader):
                pair = pair.to(self._device)
                logits = self._llm_based_model(
                    pair=pair, max_length=max_length, attention_mask=self._attention_mask
                )

                predictions += torch.nn.Sigmoid()(logits).detach().cpu()

        return predictions

    def _create_matrix(self, data: pd.DataFrame, with_label: bool) -> Any:
        """Return the correct data structure. Object that is required by the model."""
        if self._training_type in ["finetuning", "peft"]:
            dataloader, max_length = self.create_peptide_pairs_dataloader(data, with_label)
        else:
            dataloader, max_length = self.create_embeddings_pairs_dataloader(data, with_label)
        return dataloader, max_length

    # dataloader creator for probing
    def create_embeddings_pairs_dataloader(
        self, dataframe: pd.DataFrame, with_label: bool
    ) -> tuple[DataLoader, int]:
        """Creates a sequence dataset for finetuning.

        Args:
            dataframe (pd.DataFrame): dataframe containing sequence pairs and labels.
            with_label (bool): whether or not the dataset is labeled.

        Returns:
            Tuple[DataLoader, int]: Returns finetuning dataloader and length of longest seq.
        """
        # read sequences and labels from dataframe
        wild_type_sequences = list(dataframe[self._wildtype_col_name])
        mutated_sequences = list(dataframe[self._mutated_col_name])

        labels = []
        if with_label:
            labels = list(dataframe[self.label_name])

        # length of the longest sequence
        max_length = len(max(wild_type_sequences, key=len))

        # instanciate Embedding Pairs dataset
        embedding_pairs_dataset = EmbeddingsPairsDataset(
            wild_type_sequences, mutated_sequences, self._embeddings_pkl, labels
        )
        # instanciate Embedding Pairs dataloader
        embedding_pairs_dataloader = DataLoader(
            embedding_pairs_dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=self.parameters["shuffle_dataloader"],
        )
        return embedding_pairs_dataloader, max_length

    # dataloader creator for finetuning or peft
    def create_peptide_pairs_dataloader(
        self, dataframe: pd.DataFrame, with_label: bool
    ) -> tuple[DataLoader, int]:
        """Creates a sequence dataset for finetuning.

        Args:
            dataframe (pd.DataFrame): dataframe containing sequence pairs and labels.
            with_label (bool): whether or not the dataset is labeled.

        Returns:
            Tuple[DataLoader, int]: Returns finetuning dataloader and length of longest seq.
        """
        # read sequences and labels from dataframe
        wild_type_sequences = list(dataframe[self._wildtype_col_name])
        mutated_sequences = list(dataframe[self._mutated_col_name])

        labels = []
        if with_label:
            labels = list(dataframe[self.label_name])

        # crop sequences to match desired context length
        if self.parameters["context_length"]:
            if self._mutation_position_col_name and self._mutation_position_col_name in list(
                dataframe.columns
            ):
                mutation_start_positions = list(dataframe["mutation_start_position"])
                context_length = self.parameters["context_length"]
                wild_type_sequences = crop_sequences(
                    wild_type_sequences, mutation_start_positions, context_length
                )
                mutated_sequences = crop_sequences(
                    mutated_sequences, mutation_start_positions, context_length
                )

            else:
                raise ValueError(
                    "if context length is set, mutation positions should be provided"
                    "and the corresponding column should exist"
                )

        # number of tokens of the longest sequence
        max_length = max(
            [
                len(self._tokenizer.encode_plus(seq)["input_ids"])
                for seq in tqdm(wild_type_sequences + mutated_sequences)
            ]
        )

        # instanciate PeptidePairs dataset
        peptide_pairs_dataset = PeptidePairsDataset(
            wild_type_sequences, mutated_sequences, labels, self._tokenizer, max_length
        )
        # instanciate PeptidePairs dataloader
        peptide_pairs_dataloader = DataLoader(
            peptide_pairs_dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=self.parameters["shuffle_dataloader"],
            collate_fn=peptide_pairs_dataset.tokenize_batch_of_pairs,
        )
        return peptide_pairs_dataloader, max_length

    def check_training_type(self) -> None:
        """Checks if the chosen model type is valid."""
        supported_types = ["finetuning", "probing", "peft"]
        if self._training_type not in supported_types:
            raise ValueError(f"Training type '{self._training_type}' not in {supported_types}")
