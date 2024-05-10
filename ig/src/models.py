"""File that contains all the model that will be used in the experiment."""
from __future__ import annotations

import math
import sys
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import xgboost as xgb
from catboost import CatBoostClassifier
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ig.dataset.torch_dataset import MixedDataset, PeptidePairsDataset
from ig.src.evaluation import Evaluation
from ig.src.llm_based_models import FinetuningModel
from ig.src.logger import ModelLogWriter, get_logger
from ig.src.torch_utils import set_torch_reproducibility
from ig.src.utils import crop_sequences, save_as_pkl

log = get_logger("Train/model")
original_stdout = sys.__stdout__


class BaseModel(ABC):
    """Basic class Method."""

    def __init__(
        self,
        features: list[str],
        parameters: dict[str, Any],
        label_name: str,
        prediction_name: str,
        checkpoints: Path,
        other_params: dict[str, Any],
        folder_name: str,
        experiment_name: str,
        model_type: str,
        save_model: bool,
        dataset_name: str,
    ) -> None:
        """Init method."""
        self.features = features
        self.parameters = parameters
        self.label_name = label_name
        self.prediction_name = prediction_name
        self.checkpoints = checkpoints
        self.other_params = other_params
        self.folder_name = folder_name
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.shap_values = None
        self.dataset_name = dataset_name
        self.model_logger: ModelLogWriter
        self.model: Any
        self.save_model = save_model
        if self.save_model:
            self.checkpoints.parent.mkdir(parents=True, exist_ok=True)
            self.model_logger = ModelLogWriter(str(self.checkpoints.parent / "model.log"))

    @property
    def model_meta_data(self) -> dict[str, Any]:
        """Model meta data."""
        return {
            "model": self.model,
            "model_params": self.parameters,
            "features": self.features,
            "checkpoints": self.checkpoints,
        }

    @abstractmethod
    def predict(self, data: pd.DataFame, with_label: bool) -> Any:
        """Prediction method."""

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fit method."""

    def eval_model(self, data: pd.DataFrame, split_name: str, evaluator: Evaluation) -> None:
        """Eval method."""
        data = data.copy()

        data[self.prediction_name] = self.predict(data, with_label=False)
        evaluator.compute_metrics(
            data=data,
            prediction_name=self.prediction_name,
            split_name=split_name,
            dataset_name=self.dataset_name,
        )

    def generate_shap_values(self, data: pd.DataFrame) -> Any:
        """Generate shap values."""
        return shap.TreeExplainer(self.model).shap_values(data)

    @abstractmethod
    def _create_matrix(self, data: pd.DataFrame, with_label: bool) -> Any | None:
        """Return the correct data structure. Object that is required by the model."""


warnings.filterwarnings("ignore")


class XgboostModel(BaseModel):
    """Xgboost model class."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        dtrain = self._create_matrix(train_data)
        dval = self._create_matrix(val_data)

        self.model = xgb.train(
            params=self.parameters,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dval, "valid")],
            num_boost_round=self.other_params.get("num_boost_round", 20000),
            verbose_eval=self.other_params.get("verbose_eval", True),
            early_stopping_rounds=self.other_params.get("early_stopping_rounds", 20),
        )

        self.shap_values = self.generate_shap_values(train_data[self.features])

        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        dmatrix = self._create_matrix(data, with_label)
        best_iteration = self.model.best_iteration
        return self.model.predict(dmatrix, iteration_range=(0, best_iteration + 1))

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any | None:
        """Data model creation."""
        label = data[self.label_name] if with_label else None
        return xgb.DMatrix(data=data[self.features], label=label, feature_names=self.features)


class LgbmModel(BaseModel):
    """This is an implementation of lgbm model.Based on the Native Microsoft Implementation."""

    def generate_shap_values(self, data: pd.DataFrame) -> Any:
        """Generate shap values."""
        return shap.TreeExplainer(self.model).shap_values(data)[1]

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        # Load data
        true_data = train_data
        train_data = self._create_matrix(train_data)
        val_data = self._create_matrix(val_data, with_label=True)
        verbose_eval = self.other_params["verbose_eval"]
        early_stopping_rounds = self.other_params.get("early_stopping_rounds")

        # train model
        self.model = lgb.train(
            self.parameters,
            train_data,
            num_boost_round=self.other_params.get("num_boost_round"),
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(verbose_eval)],
        )
        self.shap_values = self.generate_shap_values(true_data[self.features])
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        return self.model.predict(data[self.features], num_iteration=self.model.best_iteration)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any | None:
        """Data model creation."""
        if with_label:
            return lgb.Dataset(
                data[self.features],
                label=data[self.label_name],
                feature_name=self.features,
            )

        return lgb.Dataset(data[self.features], feature_name=self.features)


class CatBoostModel(BaseModel):
    """This is an implementation of catboost model.Based on the Native Implementation."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        # create model
        model = CatBoostClassifier(**self.parameters)
        # train model
        self.model = model.fit(
            X=train_data[self.features].values,
            y=train_data[self.label_name].values,
            eval_set=(val_data[self.features].values, val_data[self.label_name].values),
            log_cout=sys.stdout,
            log_cerr=sys.stderr,
        )
        self.shap_values = self.generate_shap_values(train_data[self.features])
        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        return self.model.predict_proba(data[self.features].values)[:, 1]

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any | None:
        """Data model creation."""


class LabelPropagationModel(BaseModel):
    """Label Propagation model class."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        dtrain = self._create_matrix(train_data)
        labels = self._create_matrix(train_data, with_label=False)
        self.model = LabelPropagation(**self.parameters)
        self.model = self.model.fit(dtrain, labels)
        self.shap_values = None
        log.info(" Shap values for this model type not supported")

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        dmatrix = self._create_matrix(data, with_label)
        return self.model.predict(dmatrix)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Data model creation."""
        if with_label:
            return data[self.features].to_numpy()
        return data[self.label_name].to_numpy()


class LogisticRegressionModel(BaseModel):
    """This is an implementation of LogisticRegression model.

    Based on the Native Implementation.
    """

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:  # pylint disable=W0613
        """Fitting model."""
        self.model = LogisticRegression().fit(
            train_data[self.features], train_data[self.label_name]
        )

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data: pd.DataFrame, with_label: bool = True) -> Any:
        """Prediction method."""
        return self.model.predict_proba(data[self.features].values)[:, 1]

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Data model creation."""


class RandomForestModel(BaseModel):
    """This is an implementation of Random Forest model.

    Based on the Native Implementation.
    """

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:  # pylint disable=W0613
        """Fitting model."""
        self.model = RandomForestClassifier(**self.parameters).fit(
            train_data[self.features], train_data[self.label_name]
        )

        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    # model prediction
    def predict(self, data: pd.DataFrame, with_label: bool = True) -> np.ndarray:
        """Prediction method."""
        return self.model.predict(data[self.features])

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Data model creation."""


class SupportVectorMachineModel(BaseModel):
    """This is an implementation of support vector machine model."""

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Fitting model."""
        # create model
        model = SVC(**self.parameters)
        # train model
        self.model = model.fit(
            X=train_data[self.features].values, y=train_data[self.label_name].values
        )
        # Save model
        if self.checkpoints and self.save_model:
            save_as_pkl(self, self.checkpoints)
            self.model_logger.reset_stdout(original_stdout)

    def predict(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Prediction method."""
        return self.model.predict(data[self.features].values)

    def _create_matrix(self, data: pd.DataFrame, with_label: bool = True) -> None:
        """Data model creation."""


class LLMBasedModel(BaseModel):
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
        """Initializes LLMBasedModel.

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
        # set torch in deterministic mode (for reproducibility)
        self._tokenizer = AutoTokenizer.from_pretrained(other_params["llm_hf_model_path"])
        self._attention_mask = attention_mask
        self._training_type = other_params["training_type"]
        self.check_training_type()
        set_torch_reproducibility(self.parameters["seed"])

        if self._training_type in ["finetuning", "peft"]:
            self._llm_based_model = FinetuningModel(
                model_configuration=parameters,
                llm_hf_model_path=other_params["llm_hf_model_path"],
                is_masked_model=other_params["is_masked_model"],
                tokenizer=self._tokenizer,
                training_type=self._training_type,
            )

        elif self._training_type == "probing_model":
            # TODO: add instanciation of probing model when ready
            raise NotImplementedError

        # only save finetuned backbone to use for another task
        # instead of saving the whole LLMBasedModel
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
            epochs_iterator.set_description(f"epoch: {epoch_num}/{num_epochs}")
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
            # pickle LLMBasedModel object
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
        logits = self._llm_based_model(pair, max_length, self._attention_mask)
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
        logits = self._llm_based_model(pair, max_length, self._attention_mask)
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
                logits = self._llm_based_model(pair, max_length, self._attention_mask)
                predictions += torch.nn.Sigmoid()(logits).detach().cpu()

        return predictions

    def _create_matrix(self, data: pd.DataFrame, with_label: bool) -> Any:
        """Return the correct data structure. Object that is required by the model."""
        if self._training_type in ["finetuning", "peft"]:
            dataloader, max_length = self.create_peptide_pairs_dataloader(data, with_label)
        else:
            # TODO add EmbeddingPairs dataloader creator when ready
            raise NotImplementedError

        return dataloader, max_length

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
        supprted_types = ["finetuning", "probing", "peft"]
        if self._training_type not in supprted_types:
            raise ValueError(f"type '{self._training_type}' not in {supprted_types}")


class MixedModel(BaseModel):
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
        """Initializes LLMBasedModel.

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
            other_params["llm_hf_model_path"], trust_remote_code=True
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

        # two stages MixedModel
        if other_params["pretrained_llm_path"]:
            llm_state_dict = torch.load(other_params["pretrained_llm_path"])["model_state_dict"]
            self._llm.load_state_dict(llm_state_dict)

        if self._use_cuda:
            # send model to GPU and enable multi-gpu usage
            self._llm = torch.nn.DataParallel(self._llm).to(self._device)

        self._wildtype_col_name = other_params["wildtype_col_name"]
        self._mutated_col_name = other_params["mutated_col_name"]
        self._mutation_position_col_name = other_params["mutation_position_col_name"]

        self._metrics: dict[str, list[float]] = defaultdict(list)

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
            # pickle MixedModel object
            save_as_pkl(self, self.checkpoints)

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

    def compute_metrics(self, preds: list, labels: list) -> tuple[np.float, np.float]:
        """Computes global and positiva class accuracies.

        Args:
            preds (list): _description_
            labels (list): _description_

        Returns:
            tuple[np.float, np.float]: _description_
        """
        # compute and save metrics
        positive_indexes = [idx for idx in range(len(labels)) if labels[idx] == 1]
        epoch_pos_preds = [1 if preds[idx] == labels[idx] else 0 for idx in positive_indexes]
        # global accuracy
        acc = np.average(preds == labels)
        # positive class accuracy
        acc_pos = np.average(epoch_pos_preds)
        return acc, acc_pos

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """Train MixedModel model.

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
        train_preds = np.round(
            self.ml_model.predict(dtrain, iteration_range=(0, best_iteration + 1))
        )
        val_preds = np.round(self.ml_model.predict(dval, iteration_range=(0, best_iteration + 1)))

        train_acc, train_acc_pos = self.compute_metrics(train_preds, train_labels)
        val_acc, val_acc_pos = self.compute_metrics(val_preds, val_labels)
        self._metrics["train_acc"] = train_acc
        self._metrics["val_acc"] = val_acc
        self._metrics["train_acc_pos"] = train_acc_pos
        self._metrics["val_acc_pos"] = val_acc_pos

    def predict(self, data: pd.DataFame, with_label: bool) -> Any:
        """Prediction method."""
        inference_dataloader, _ = self._create_matrix(data, with_label)
        combined_features, labels = self.compute_combined_features(inference_dataloader)
        dinference = xgb.DMatrix(data=combined_features, label=labels)
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

    def log_metrics(self) -> None:
        """Log epoch metrics."""
        log.info(
            f"Metrics: Train Acc: {self._metrics['train_acc']: .3f} \
            | Train Acc Positives: {self._metrics['train_acc_pos']: .3f} \
            | Val Acc: {self._metrics['val_acc']: .3f} \
            | Val Acc Positives: {self._metrics['val_acc_pos']: .3f}"
        )


BaseModelType = Union[
    XgboostModel,
    LgbmModel,
    CatBoostClassifier,
    LabelPropagationModel,
    LogisticRegressionModel,
    RandomForestModel,
    SupportVectorMachineModel,
    LLMBasedModel,
    MixedModel,
]
