"""File that contains all the model that will be used in the experiment."""
from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ig.dataset.torch_dataset import PeptidePairsDataset
from ig.src.evaluation import Evaluation
from ig.src.llm_based_models import FinetuningModel
from ig.src.logger import ModelLogWriter, get_logger
from ig.src.utils import save_as_pkl

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
        tokenizer: AutoTokenizer,
        checkpoints: Path,
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
            features=["llm_embeddings"],
            label_name="label",
            prediction_name="prediction",
            other_params={},
            folder_name=folder_name,
            model_type=model_type,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            checkpoints=checkpoints,
            save_model=save_model,
        )
        self._tokenizer = tokenizer
        self._attention_mask = attention_mask
        self.check_model_type()
        if model_type in ["finetuning_model", "peft_model"]:
            self._llm_based_model = FinetuningModel(
                model_configuration=parameters,
                tokenizer=tokenizer,
                model_type=model_type,
            )

        elif model_type == "probing_model":
            # TODO: add instanciation of probing model when ready
            raise NotImplementedError

        self._use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer = Adam(self._llm_based_model.parameters(), lr=self.parameters["lr"])
        self._metrics: dict = {
            "train_loss_per_step": [],
            "train_loss_per_epoch": [],
            "train_acc": [],
            "val_loss_per_step": [],
            "val_loss_per_epoch": [],
            "val_acc": [],
        }

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
        max_length = max([max_length_train, max_length_val])

        if self._use_cuda:
            # send model to GPU and enable multi-gpu usage
            self._llm_based_model = torch.nn.DataParallel(self._llm_based_model).to(self._device)

        num_epochs = self.parameters["num_epochs"]
        epochs_iterator = tqdm(range(num_epochs))
        for epoch_num in epochs_iterator:
            epochs_iterator.set_description(f"epoch: {epoch_num}/{num_epochs}")
            self.train_one_epoch(train_dataloader, max_length)
            self.validate_one_epoch(val_dataloader, max_length)
            self.log_metrics(epoch_num)

    def train_one_epoch(self, train_dataloader: DataLoader, max_length: int) -> None:
        """Train model for one epoch.

        Args:
            train_dataloader (DataLoader): _description_
            max_length (int): _description_
        """
        self._llm_based_model.train()
        epoch_preds = []
        epoch_labels = []
        for pair, label in tqdm(train_dataloader):
            batch_loss, batch_preds, batch_labels = self.train_one_step(pair, label, max_length)
            epoch_preds += batch_preds
            epoch_labels += batch_labels

        epoch_preds = torch.Tensor(epoch_preds)
        self._metrics["train_acc"].append(
            np.average(np.array(torch.round(epoch_preds)) == epoch_labels)
        )
        # loss at the last step
        self._metrics["train_loss_per_epoch"].append(batch_loss)

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
        prediction = self._llm_based_model(pair, max_length, self._attention_mask)

        # calculate step loss
        batch_loss = self._criterion(prediction.squeeze(1), label)
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
            for pair, label in tqdm(val_dataloader):
                batch_loss, batch_preds, batch_labels = self.validate_one_step(
                    pair, label, max_length
                )
            epoch_preds += batch_preds
            epoch_labels += batch_labels

        epoch_preds = torch.Tensor(epoch_preds)
        self._metrics["val_acc"].append(
            np.average(np.array(torch.round(epoch_preds)) == epoch_labels)
        )
        # loss at the last step
        self._metrics["val_loss_per_epoch"].append(batch_loss)

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
        prediction = self._llm_based_model(pair, max_length, self._attention_mask)

        # calculate step loss
        batch_loss = self._criterion(prediction.squeeze(1), label)
        self._metrics["val_loss_per_step"].append(batch_loss.item())

        return batch_loss, prediction, label

    def log_metrics(self, epoch_num: int) -> None:
        """Log epoch metrics.

        Args:
            epoch_num (_type_): _description_
        """
        epoch_train_loss = self._metrics["train_loss_per_epoch"][epoch_num]
        epoch_train_acc = self._metrics["train_acc"][epoch_num]
        epoch_val_loss = self._metrics["val_loss_per_epoch"][epoch_num]
        epoch_val_acc = self._metrics["val_acc"][epoch_num]

        log.info(
            f"Epoch: {epoch_num + 1} | Train Loss: {epoch_train_loss: .3f} \
            | Train Accuracy: {epoch_train_acc: .3f} \
            | Val Loss: {epoch_val_loss: .3f} \
            | Val Accuracy: {epoch_val_acc: .3f}"
        )

    def predict(self, data: pd.DataFame, with_label: bool) -> Any:
        """Prediction method."""
        inference_dataloader, max_length = self._create_matrix(data, with_label)

        if self._use_cuda:
            # send model to GPU and enable multi-gpu usage
            self._llm_based_model = torch.nn.DataParallel(self._llm_based_model).to(self._device)
        log.info(f"Started inference using: {self._device}")

        predictions = []
        labels = []
        for pair, label in tqdm(inference_dataloader):
            logits = self._llm_based_model(pair, max_length)
            predictions.append(torch.nn.sigmoid(logits))
            labels.append(label)

        if not with_label:
            log.info("The returned lables are randomly generated")

        return predictions, labels

    def _create_matrix(self, data: pd.DataFrame, with_label: bool) -> Any:
        """Return the correct data structure. Object that is required by the model."""
        if self.model_type in ["finetuning_model", "peft_model"]:
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

        Returns:
            Tuple[DataLoader, int]: Returns finetuning dataloader and length of longest seq.
        """
        # read sequences and labels from dataframe
        wild_type_sequences = dataframe["wild_type"]
        mutated_sequences = dataframe["mutated"]
        if with_label:
            labels = dataframe["label"]
        else:
            # if the dataset is unlabled, generate random binary labels
            labels = np.random.randint(2, size=len(wild_type_sequences))

        # number of tokens of the longest sequence
        max_length = (
            max([len(seq) for seq in wild_type_sequences + mutated_sequences])
            // self.parameters["k_for_kmers"]
            + 1  # account for the prepended cls token
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

    def check_model_type(self) -> None:
        """Checks if the chosen model type is valid."""
        supprted_types = ["finetuning_model", "probing_model", "peft_model"]
        if self.model_type not in supprted_types:
            raise ValueError(f"type '{self.model_type}' not in {supprted_types}")


BaseModelType = Union[
    XgboostModel,
    LgbmModel,
    CatBoostClassifier,
    LabelPropagationModel,
    LogisticRegressionModel,
    RandomForestModel,
    SupportVectorMachineModel,
]

TrainSingleModelType = BaseModelType
TrainKfoldType = Dict[str, BaseModelType]
TrainDoubleKfold = Dict[str, BaseModelType]
TrainMultiSeedKfold = Dict[str, Dict[str, BaseModelType]]
TrainTuneType = Dict[str, Any]
TrainType = Union[
    TrainSingleModelType,
    TrainKfoldType,
    TrainDoubleKfold,
    TrainMultiSeedKfold,
    TrainTuneType,
]
