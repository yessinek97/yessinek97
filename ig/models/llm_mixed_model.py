"""File that contains the LLM Mixed model that will be used in the experiment."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ig.dataset.torch_dataset import MixedDataset
from ig.models.base_model import BaseModel, log
from ig.utils.general import crop_sequences
from ig.utils.io import save_as_pkl
from ig.utils.torch import compute_f1_score, compute_roc_score, compute_top_k, round_probs


class LLMMixedModel(BaseModel):
    """Architecture that combines the LLM based model with an ML algorithm."""

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
        save_model: bool = False,
    ):
        """Initializes LLMMixedModel.

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
        self._tokenizer = AutoTokenizer.from_pretrained(other_params["llm_hf_model_path"])
        self._use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")

        self._llm = AutoModelForMaskedLM.from_pretrained(
            other_params["llm_hf_model_path"], trust_remote_code=True, output_hidden_states=True
        )
        if other_params["training_type"] == "peft":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=self.parameters["lora_config"]["r"],
                lora_alpha=self.parameters["lora_config"]["lora_alpha"],
                lora_dropout=self.parameters["lora_config"]["lora_dropout"],
                target_modules=self.parameters["lora_config"]["target_modules"],
            )
            self._llm = get_peft_model(self._llm, peft_config)

        self._prob_threshold = parameters["threshold"]

        # two stages LLMMixedModel
        if other_params["pretrained_llm_path"]:
            llm_state_dict = torch.load(other_params["pretrained_llm_path"])["model_state_dict"]
            self._llm.load_state_dict(llm_state_dict)

        if self._use_cuda:
            # send model to GPU and enable multi-gpu usage
            self._llm = torch.nn.DataParallel(self._llm).to(self._device)

        self._wildtype_col_name = other_params["wildtype_col_name"]
        self._mutated_col_name = other_params["mutated_col_name"]
        self._mutation_position_col_name = other_params["mutation_position_col_name"]

        self._metrics: dict[str, float] = defaultdict(float)

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
    ) -> None:
        """Fit method."""
        log.info(f"Started training: {self.model_type} | using: {self._device}")

        train_dataloader, max_length_train = self._create_matrix(train_data, with_label=True)
        val_dataloader, max_length_val = self._create_matrix(val_data, with_label=True)
        # number of tokens of the longest sequence
        log.info(
            f"Maximum length in train: {max_length_train} - "
            f"Maximum length in validation: {max_length_val}"
        )
        self.train(train_dataloader, val_dataloader)
        self.log_metrics()

        if self.save_model:
            # pickle LLMMixedModel object
            save_as_pkl(self, self.checkpoints)

    def compute_metrics(self, probs: torch.Tensor, labels: torch.Tensor, stage: str) -> None:
        """Computes topK, F1 and ROC scores."""
        preds = round_probs(probs, self._prob_threshold)
        self._metrics[f"{stage}_accuracy"] = np.average(preds == labels)
        self._metrics[f"{stage}_top_k"] = compute_top_k(probs, labels)
        self._metrics[f"{stage}_f1"] = compute_f1_score(probs, labels, self._prob_threshold)
        self._metrics[f"{stage}_roc"] = compute_roc_score(probs, labels)

    def log_metrics(self) -> None:
        """Log epoch metrics.

        Args:
            epoch_num (_type_): _description_
        """
        train_acc = self._metrics["train_accuracy"]
        val_acc = self._metrics["val_accuracy"]

        train_top_k = self._metrics["train_top_k"]
        val_top_k = self._metrics["val_top_k"]

        train_f1 = self._metrics["train_f1"]
        val_f1 = self._metrics["val_f1"]

        train_roc = self._metrics["train_roc"]
        val_roc = self._metrics["val_roc"]

        log.info(
            f"\n metrics:\n \
            | Train Accuracy: {train_acc: .3f} | Val Accuracy: {val_acc: .3f} \n \
            | Train topK: {train_top_k: .3f} | Val topK: {val_top_k: .3f} \n \
            | Train F1-score: {train_f1: .3f} | Val F1-score: {val_f1: .3f} \n \
            | Train ROC-score: {train_roc: .3f} | Val ROC-score: {val_roc: .3f} \n \
            "
        )

    def compute_combined_features(self, dataloader: DataLoader) -> tuple[np.array, list]:
        """Computes embeddings and combines them with tabular features.

        Args:
            dataloader (_type_): _description_

        Returns:
            Tuple[np.array, list]: _description_
        """
        pair_embeddings, pair_features, pair_labels = [], [], []
        iterator = tqdm(dataloader)
        # compute embeddings
        for pair, features, label in iterator:
            pair_embeddings += list(pair.detach().cpu().numpy())
            pair_features += features
            pair_labels += list(label)

        # combine embeddings with tabular features
        combined_features = np.array(
            [
                np.concatenate((embeddings, features))
                for embeddings, features in zip(pair_embeddings, pair_features)
            ]
        )
        return combined_features, pair_labels

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """Train LLMMixedModel model.

        Args:
            train_dataloader (DataLoader): _description_
            val_dataloader (_type_): _description_
        """
        log.info("Preparing combined features and labels")
        train_combined_features, train_labels = self.compute_combined_features(train_dataloader)
        val_combined_features, val_labels = self.compute_combined_features(val_dataloader)

        dtrain = xgb.DMatrix(data=train_combined_features, label=train_labels)
        dval = xgb.DMatrix(data=val_combined_features, label=val_labels)

        log.info("fitting ML model")
        self.ml_model = xgb.train(
            params=self.parameters["ml_model_params"],
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dval, "valid")],
            num_boost_round=self.other_params.get("num_boost_round", 20000),
            verbose_eval=self.other_params.get("verbose_eval", True),
            early_stopping_rounds=self.other_params.get("early_stopping_rounds", 20),
        )

        best_iteration = self.ml_model.best_iteration
        train_probs = self.ml_model.predict(dtrain, iteration_range=(0, best_iteration + 1))
        val_probs = self.ml_model.predict(dval, iteration_range=(0, best_iteration + 1))
        # using torch metrics to make sure different models are fairly evaluated
        self.compute_metrics(torch.Tensor(train_probs), torch.Tensor(train_labels), "train")
        self.compute_metrics(torch.Tensor(val_probs), torch.Tensor(val_labels), "val")

    def predict(self, data: pd.DataFame, with_label: bool) -> Any:
        """Prediction method."""
        inference_dataloader, _ = self._create_matrix(data, with_label)
        combined_features, _ = self.compute_combined_features(inference_dataloader)
        dinference = xgb.DMatrix(data=combined_features, label=None)
        best_iteration = self.ml_model.best_iteration
        predictions = self.ml_model.predict(dinference, iteration_range=(0, best_iteration + 1))
        return predictions

    def _create_matrix(self, data: pd.DataFrame, with_label: bool) -> Any:
        """Return the correct data structure. Object that is required by the model."""
        return self.create_mixed_dataloader(data, with_label)

    # dataloader creator for mixed approach
    def create_mixed_dataloader(
        self, dataframe: pd.DataFrame, with_label: bool
    ) -> tuple[DataLoader, int]:
        """Creates mixed dataset for mixed approach."""
        # read sequences and labels from dataframe
        wild_type_sequences = list(dataframe[self._wildtype_col_name])
        mutated_sequences = list(dataframe[self._mutated_col_name])
        tabular_features_dict = {feature: list(dataframe[feature]) for feature in self.features}

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
        # instanciate Mixed dataset
        mixed_dataset = MixedDataset(
            wild_type_sequences,
            mutated_sequences,
            tabular_features_dict,
            labels,
            self._llm,
            self._tokenizer,
            self._device,
            max_length,
        )
        # instanciate Mixed dataloader
        mixed_dataloader = DataLoader(
            mixed_dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=self.parameters["shuffle_dataloader"],
            collate_fn=mixed_dataset.compute_batch_mutation_embeddings,
        )

        return mixed_dataloader, max_length
