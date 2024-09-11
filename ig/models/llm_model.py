"""File that contains all the deep learning models."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ig.dataset.torch_dataset import EmbeddingsPairsDataset, PeptidePairsDataset
from ig.models.base_model import BaseModel, log
from ig.models.torch_based_models import FinetuningModel, FocusedFinetuningModel, ProbingModel
from ig.utils.embedding import load_embedding_file
from ig.utils.general import crop_sequences
from ig.utils.io import save_as_pkl
from ig.utils.torch import (
    TOKENIZER_SOURCES,
    compute_f1_score,
    compute_roc_score,
    compute_top_k,
    create_scheduler,
    round_probs,
    set_torch_reproducibility,
)


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
        save_model: bool = True,
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
            save_model (bool, optional): _description_. Defaults to True.

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
            tokenizer_source = TOKENIZER_SOURCES[other_params["tokenizer_source"]]
            self._tokenizer = tokenizer_source.from_pretrained(other_params["llm_hf_model_path"])
            llm_model_params_dict = {
                "model_configuration": parameters,
                "llm_hf_model_path": other_params["llm_hf_model_path"],
                "model_source": other_params["model_source"],
                "tokenizer": self._tokenizer,
                "training_type": self._training_type,
            }
            # Initialize model that uses the chosen aggregation method
            if self.parameters["aggregation"] == "mut_token":
                self._llm_based_model = FocusedFinetuningModel(**llm_model_params_dict)
            else:
                self._llm_based_model = FinetuningModel(**llm_model_params_dict)

        elif self._training_type == "probing":
            self._llm_based_model = ProbingModel(model_configuration=parameters)
            # Load embeddings from pickle file
            self._embeddings_pkl = load_embedding_file(other_params["emb_file_path"])

        # only save finetuned backbone to use for another task
        # instead of saving the whole LLMModel
        self._save_llm_only = other_params["save_llm_only"]

        # send model to GPU and enable multi-gpu usage
        self._model_is_data_parallel = False
        self._use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        if self._use_cuda & (torch.cuda.device_count() > 1):
            self._llm_based_model = torch.nn.DataParallel(self._llm_based_model)
            self._model_is_data_parallel = True
        self._llm_based_model.to(self._device)

        self._wildtype_col_name = other_params["wildtype_col_name"]
        self._mutated_col_name = other_params["mutated_col_name"]
        self._mutation_position_col_name = other_params["mutation_position_col_name"]

        self._prob_threshold = parameters["threshold"]

        self._optimizer = Adam(self._llm_based_model.parameters(), lr=self.parameters["lr"])
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self.backward_fn = self.create_backward_pass()

        self._metrics: dict[str, list[float]] = defaultdict(list)

        # Get early stopping params
        self._val_frequency = float(self.parameters["val_frequency"])  # should be in ]0, 1]
        self._max_patience = int(self.parameters["early_stop_patience"])
        self._early_stop_metric = self.parameters["early_stop_metric"]
        self._is_early_stop = False
        self._patience_counter = 0
        if self.parameters["early_stop_metric_max"]:
            self._best_metric = 0.0
        else:
            self._best_metric = np.inf

        self._shuffle_dataloader = self.parameters["shuffle_dataloader"]

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
            \t\tnumber of trainable parameter: {num_trainable_params}\n"
        )

        train_dataloader, max_length_train = self._create_matrix(
            train_data, with_label=True, shuffle=self._shuffle_dataloader
        )
        val_dataloader, max_length_val = self._create_matrix(
            val_data, with_label=True, shuffle=False
        )

        # number of training steps (batchs) after which we will do a validation round
        self._one_round_size = int(len(train_dataloader) * self._val_frequency)
        log.info(
            "There are %s steps in one epoch,\n \
            \t\tWith a frequency of %s, a validation round will be triggered after each %s steps,\n\
            \t\twhich results in %s validation rounds per epoch!\n",
            len(train_dataloader),
            self._val_frequency,
            self._one_round_size,
            len(train_dataloader) / self._one_round_size,
        )
        self._steps_counter = 0

        num_epochs = self.parameters["num_epochs"]
        if self.other_params["lr_scheduler_config"]["use_scheduler"]:
            self._scheduler = create_scheduler(
                self.other_params["lr_scheduler_config"],
                num_epochs,
                len(train_dataloader),
                self._optimizer,
            )

        # number of tokens of the longest sequence
        log.info(
            f"Maximum length in train: {max_length_train} - "
            f"Maximum length in validation: {max_length_val}\n"
        )

        epochs_iterator = range(num_epochs)
        for epoch_num in epochs_iterator:
            log.info(f"Starting Epoch: {epoch_num+1}/{num_epochs}\n")
            self.train_one_epoch(
                train_dataloader,
                max_length_train,
                val_dataloader,
                max_length_val,
            )

            # Break out of training loop if early stopping is reached
            if self._is_early_stop:
                log.info(
                    "Early stopping %s/%s is reached after %s total training steps ! \n",
                    self._patience_counter,
                    self._max_patience,
                    self._steps_counter,
                )
                break

        # Load the best model weights from .pt checkpoint for later evaluation step
        torch_best_model = torch.load(str(self.checkpoints).replace("model.pkl", "torch_model.pt"))
        # Replace the current model state_dict with the best one
        self._llm_based_model.load_state_dict(torch_best_model["model_state_dict"])

        if self.save_model:
            # Save final model (.pkl + .pt) with best weights
            self.save_final_checkpoint()

    def train_one_epoch(
        self,
        train_dataloader: DataLoader,
        train_max_length: int,
        val_dataloader: DataLoader,
        val_max_length: int,
    ) -> None:
        """Train model for one epoch.

        Args:
            train_dataloader (DataLoader): _description_
            train_max_length (int): _description_
            val_dataloader (DataLoader): _description_
            val_max_length (int): _description_
        """
        self._llm_based_model.train()
        epoch_probs = []
        epoch_labels = []
        epoch_loss = 0.0
        iterator = tqdm(train_dataloader)
        for pair, label in iterator:
            batch_loss, batch_probs, batch_labels = self.train_one_step(
                pair, label, train_max_length
            )
            self._steps_counter += 1
            iterator.set_postfix({"Loss": batch_loss.item()})
            epoch_probs += batch_probs
            epoch_labels += batch_labels
            epoch_loss += batch_loss
            # Run validation If steps_counter reachs one_round_size (val_frequency)
            if self._steps_counter % self._one_round_size == 0:
                train_round_probs = torch.Tensor(epoch_probs).detach().cpu()
                train_round_labels = torch.Tensor(epoch_labels).detach().cpu()
                train_round_loss = epoch_loss / len(train_dataloader)
                self.compute_round_metrics(
                    train_round_probs, train_round_labels, train_round_loss, "train_round"
                )
                self.validate_one_round(val_dataloader, val_max_length)

                # Check whether a new best score is reached, Save the model and check if early stopping condition is met
                self.early_stopping()

            if self._is_early_stop:
                break

        epoch_probs = torch.Tensor(epoch_probs).detach().cpu()
        epoch_labels = torch.Tensor(epoch_labels).detach().cpu()
        epoch_loss = epoch_loss / len(train_dataloader)
        self.compute_round_metrics(epoch_probs, epoch_labels, epoch_loss, "train_epoch")
        log.info("Epoch Metrics:\n")
        log.info(self.log_metrics("train_epoch"))

    def forward_pass(
        self, pair: torch.Tensor, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass.

        Args:
            pair (torch.Tensor): _description_
            max_lenvgth (int): _description_

        Returns:
            torch.Tensor: returns rounded predictions and logits
        """
        # forward pass
        logits = self._llm_based_model(
            pair=pair, max_length=max_length, attention_mask=self._attention_mask
        )
        probs = torch.nn.Sigmoid()(logits)

        return logits, probs

    def create_backward_pass(self) -> Callable:
        """Returns the a backward function depending on using or not using a scheduler.

        Returns:
            callable: backward function
        """
        return (
            self.backward_pass
            if not self.other_params["lr_scheduler_config"]["use_scheduler"]
            else self.backward_pass_with_scheduler
        )

    def backward_pass(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Performs a backward pass.

        Args:
            logits (torch.Tensor): _description_
            label (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # calculate step loss
        batch_loss = self._criterion(logits.squeeze(1), label)
        # clear last pass gradient error
        self._optimizer.zero_grad()
        # back propagate loss and update gradient
        batch_loss.backward()
        self._optimizer.step()
        return batch_loss

    def backward_pass_with_scheduler(
        self, logits: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """Performs backward pass and updates scheduler.

        Args:
            logits (torch.Tensor): _description_
            label (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        self._scheduler.step()
        batch_loss = self.backward_pass(logits, label)
        return batch_loss

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
        # perform forward pass
        logits, probs = self.forward_pass(pair, max_length)
        # perform backward pass
        batch_loss = self.backward_fn(logits, label)
        self._metrics["train_loss_per_step"].append(batch_loss.item())

        return batch_loss, probs, label

    def validate_one_round(self, val_dataloader: DataLoader, max_length: int) -> None:
        """Validate model for one round.

        Args:
            val_dataloader (DataLoader): _description_
            max_length (int): _description_


        Returns:
            _type_: _description_
        """
        self._llm_based_model.eval()
        with torch.no_grad():
            val_probs = []
            val_labels = []
            val_loss = 0.0
            for pair, label in val_dataloader:
                batch_loss, batch_probs, batch_labels = self.validate_one_step(
                    pair, label, max_length
                )
                val_probs += batch_probs
                val_labels += batch_labels
                val_loss += batch_loss

        val_probs = torch.Tensor(val_probs).detach().cpu()
        val_labels = torch.Tensor(val_labels).detach().cpu()
        val_loss = val_loss / len(val_dataloader)
        self.compute_round_metrics(val_probs, val_labels, val_loss, "val_round")
        self._llm_based_model.train()

    def validate_one_step(
        self, pair: torch.Tensor, label: torch.Tensor, max_length: int
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Make one forward pass.

        Args:
            pair (torch.Tensor): _description_
            label (torch.Tensor): _description_
            max_length (int): _description_
        Returns:
            torch.Tensor: returns probabilities and labels
        """
        pair = pair.to(self._device)
        label = label.to(self._device)
        logits, probs = self.forward_pass(pair, max_length)
        # calculate step loss
        batch_loss = self._criterion(logits.squeeze(1), label)
        self._metrics["val_loss_per_step"].append(batch_loss.item())

        return batch_loss, probs, label

    def compute_round_metrics(
        self, probs: list[float], round_labels: list[int], loss: float, stage: str
    ) -> None:
        """Computes round metrics and appends them to metrics dict."""
        round_preds = round_probs(probs, self._prob_threshold)
        self._metrics[f"{stage}_accuracy"].append(np.average(round_preds == round_labels))
        self._metrics[f"{stage}_top_k"].append(compute_top_k(probs, round_labels))
        self._metrics[f"{stage}_f1"].append(
            compute_f1_score(probs, round_labels, self._prob_threshold)
        )
        self._metrics[f"{stage}_roc"].append(compute_roc_score(probs, round_labels))
        self._metrics[f"{stage}_loss"].append(loss)

    def log_metrics(self, stage: str) -> str:
        """Log round metrics.

        Args:
            stage (_type_): _description_
        Returns:
            str: log message

        """
        acc = self._metrics[f"{stage}_accuracy"][-1]
        top_k = self._metrics[f"{stage}_top_k"][-1]
        f1 = self._metrics[f"{stage}_f1"][-1]
        roc = self._metrics[f"{stage}_roc"][-1]
        loss = self._metrics[f"{stage}_loss"][-1]

        round_num = len(self._metrics[f"{stage}_accuracy"])
        message = (
            f"{stage} {round_num} metrics : "
            f" | Accuracy: {acc: .3f}"
            f" | topK: {top_k: .3f}"
            f" | F1-score: {f1: .3f}"
            f" | ROC-score: {roc: .3f}"
            f" | log-loss: {loss: .3f}\n"
        )

        if stage == "val_round":
            current_metric = self._metrics[f"val_round_{self._early_stop_metric}"][-1]
            message += (
                "\t\t\tEarly_stop_metric: "
                f"Current val_round_{self._early_stop_metric} : {current_metric: .3f} | "
                f"Best val_round_{self._early_stop_metric} : {self._best_metric: .3f}\n"
            )
        return message

    def predict(self, data: pd.DataFame, with_label: bool) -> Any:
        """Prediction method."""
        inference_dataloader, max_length = self._create_matrix(data, with_label, shuffle=False)

        self._llm_based_model.eval()

        predictions = []
        with torch.no_grad():
            for pair, _ in tqdm(inference_dataloader):
                pair = pair.to(self._device)
                _, step_predictions = self.forward_pass(pair, max_length)
                predictions += step_predictions.detach().cpu()

        return predictions

    def _create_matrix(self, data: pd.DataFrame, with_label: bool, **kargs: bool) -> Any:
        """Return the correct data structure. Object that is required by the model."""
        if self._training_type in ["finetuning", "peft"]:
            dataloader, max_length = self.create_peptide_pairs_dataloader(
                data, with_label, kargs["shuffle"]
            )
        else:
            dataloader, max_length = self.create_embeddings_pairs_dataloader(
                data, with_label, kargs["shuffle"]
            )
        return dataloader, max_length

    # dataloader creator for probing
    def create_embeddings_pairs_dataloader(
        self, dataframe: pd.DataFrame, with_label: bool, shuffle: bool
    ) -> tuple[DataLoader, int]:
        """Creates an embeddings dataset for probing.

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
            shuffle=shuffle,
            num_workers=self.parameters["num_workers"],
        )
        return embedding_pairs_dataloader, max_length

    # dataloader creator for finetuning or peft
    def create_peptide_pairs_dataloader(
        self, dataframe: pd.DataFrame, with_label: bool, shuffle: bool
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
                mutation_start_positions = list(dataframe[self._mutation_position_col_name])
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
            shuffle=shuffle,
            num_workers=self.parameters["num_workers"],
            collate_fn=peptide_pairs_dataset.tokenize_batch_of_pairs,
        )
        return peptide_pairs_dataloader, max_length

    def check_training_type(self) -> None:
        """Checks if the chosen model type is valid."""
        supported_types = ["finetuning", "probing", "peft"]
        if self._training_type not in supported_types:
            raise ValueError(f"Training type '{self._training_type}' not in {supported_types}")

    def early_stopping(self) -> None:
        """Save best model and trigger the early stopping if the validation score didn't improve after X steps."""
        current_metric = self._metrics[f"val_round_{self._early_stop_metric}"][-1]
        log.info(self.log_metrics("train_round"))
        log.info(self.log_metrics("val_round"))
        cond_1 = (current_metric > self._best_metric) & self.parameters["early_stop_metric_max"]
        cond_2 = (current_metric < self._best_metric) & (
            not self.parameters["early_stop_metric_max"]
        )
        if cond_1 or cond_2:
            # Save new best metric
            self._best_metric = current_metric
            self._patience_counter = 0

            # Keep the best model weights
            self.save_current_checkpoint()

            # Set early stopping to False
            self._is_early_stop = False

        else:
            self._patience_counter += 1
            log.info(f"Early stopping counter {self._patience_counter} / {self._max_patience}\n")
            if self._patience_counter >= self._max_patience:
                # Set early stopping to True
                self._is_early_stop = True
                log.info("Early stopping is Reached !")

    def save_current_checkpoint(self) -> None:
        """Temporarily save the entire current best model in .pt file."""
        # Best model to be saved
        dict_to_save = {
            "model_state_dict": self._llm_based_model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "metrics": self._metrics,
        }
        # Save the current best model to temporary .pt file
        torch.save(dict_to_save, str(self.checkpoints).replace("model.pkl", "torch_model.pt"))

    def save_final_checkpoint(self) -> None:
        """Save the final best model checkpoints (.pkl + .pt) with option to only save the LLM backbone."""
        # Select the torch model to be saved in .pt checkpoint
        # _save_llm_only must be true for later use in LLMMixedModel
        if self._save_llm_only:
            llm_state_dict = (
                self._llm_based_model.module.llm.state_dict()
                if self._model_is_data_parallel
                else self._llm_based_model.llm.state_dict()
            )
            dict_to_save = {"model_state_dict": llm_state_dict}
        else:
            model_state_dict = (
                self._llm_based_model.module.state_dict()
                if self._model_is_data_parallel
                else self._llm_based_model.state_dict()
            )
            dict_to_save = {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self._optimizer.state_dict(),
                "metrics": self._metrics,
            }
        # Save final torch_model.pt
        torch.save(dict_to_save, str(self.checkpoints).replace("model.pkl", "torch_model.pt"))

        log.info(f"Saving final best checkpoint to: {self.checkpoints} \n\n")
        # pickle the LLMBasedModel object to be used in IG framework for model evaluation, inference, ...
        save_as_pkl(self, self.checkpoints)
