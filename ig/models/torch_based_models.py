"""Definition of the TorchBaseModel class."""
from typing import Any, Dict, Union

import torch
from multimolecule import RnaTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer

from ig.utils.torch import LLM_MODEL_SOURCES, find_mut_token


class TorchBaseModel(torch.nn.Module):
    """Classifies the aggregated embeddings of the mutated and wild type sequences.

    Args:
        Super class (torch.nn.Module): _description_
    """

    def __init__(
        self,
        model_configuration: Dict[str, Any],
    ):
        """Initializes the model object.

        Args:
            model_configuration (Dict[str, Any]): dict containing model configuration
        """
        super().__init__()
        self._model_configuration = model_configuration

        mlp_layer_sizes = self._model_configuration["mlp_layer_sizes"]
        if not mlp_layer_sizes:
            mlp_layer_sizes = [self._model_configuration["embed_dim"], 1]

        else:
            mlp_layer_sizes = [self._model_configuration["embed_dim"]] + mlp_layer_sizes + [1]

        self._mlp_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(mlp_layer_sizes[idx], mlp_layer_sizes[idx + 1])
                for idx in range(len(mlp_layer_sizes) - 1)
            ]
        )
        self._relu = torch.nn.ReLU()
        self._sigmoid = torch.nn.Sigmoid()


class FinetuningModel(TorchBaseModel):
    """Classifies the aggregated embeddings of the mutated and wild type sequences.

    Args:
        Super class (torch.nn.Module): _description_
    """

    def __init__(
        self,
        model_configuration: Dict[str, Any],
        llm_hf_model_path: str,
        model_source: str,
        training_type: str,
        tokenizer: Union[AutoTokenizer, RnaTokenizer],
    ):
        """Initializes the model object.

        Args:
            model_configuration (Dict[str, Any]): dict containing model configuration
            llm_hf_model_path (str): path to HuggingFace model
            model_source (str): HF class name under which the model is implemented
            training_type: training strategy, peft, finetuning or probing
            tokenizer (AutoTokenizer): tokenize sequences according to the chosen LLM
        """
        super().__init__(model_configuration=model_configuration)

        self._llm_hf_model_path = llm_hf_model_path
        self._model_source = LLM_MODEL_SOURCES[model_source]
        self.llm = self._model_source.from_pretrained(
            self._llm_hf_model_path, trust_remote_code=True, output_hidden_states=True
        )
        self._tokenizer = tokenizer
        if training_type == "peft":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=self._model_configuration["lora_config"]["r"],
                lora_alpha=self._model_configuration["lora_config"]["lora_alpha"],
                lora_dropout=self._model_configuration["lora_config"]["lora_dropout"],
                target_modules=self._model_configuration["lora_config"]["target_modules"],
            )
            self.llm = get_peft_model(self.llm, peft_config)

    def forward(self, **kwargs: Dict) -> torch.Tensor:
        """Definition of the model forward pass.

        Returns:
            torch.Tensor: binary immune response prediction
        """
        x: torch.Tensor = kwargs["pair"]
        attention_mask: torch.Tensor = kwargs["attention_mask"]
        max_length = kwargs["max_length"]

        batch_size = x.shape[0]

        if not attention_mask:
            attention_mask = x != self._tokenizer.cls_token_id

        # batch embeddings of the (wild_type, mutated) sequence pairs
        x = self.llm(
            x.view(batch_size * 2, -1),  # flatten along batch to compute all embeddings in one pass
            attention_mask=attention_mask.view(batch_size * 2, -1),
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )["hidden_states"][-1]

        # group embeddings along batch
        x = torch.reshape(x, (batch_size, 2, max_length, -1))

        # mutation embedding given by the difference between the mutated and wild type
        mutation_embeddings = x[:, 0, :, :] - x[:, 1, :, :]

        # calculate attention mask of the mutation embeddings in a way that positions
        # that are masked on either of the wild type or mutated sequences,
        # is masked on the final (mutation) sequence
        mutation_mask = torch.unsqueeze(
            torch.mul(attention_mask[:, 0, :], attention_mask[:, 1, :]),
            dim=-1,
        )

        aggregated_embedding = torch.sum(mutation_mask * mutation_embeddings, axis=-2) / torch.sum(
            mutation_mask
        )

        # pass aggregated embedding through MLP layers
        for layer in self._mlp_layers[:-1]:
            aggregated_embedding = layer(aggregated_embedding)
            aggregated_embedding = self._relu(aggregated_embedding)

        logits = self._mlp_layers[-1](aggregated_embedding)

        return logits


class FocusedFinetuningModel(FinetuningModel):
    """Classifies the aggregated embeddings of the mutated and wild type sequences. Only Uses the embedding of the token containing the mutation.

    Args:
        Super class (torch.nn.Module): _description_
    """

    def __init__(
        self,
        model_configuration: Dict[str, Any],
        model_source: str,
        llm_hf_model_path: str,
        training_type: str,
        tokenizer: Union[AutoTokenizer, RnaTokenizer],
    ):
        """Initializes the model object.

        Args:
            model_configuration (Dict[str, Any]): dict containing model configuration
            llm_hf_model_path (str): path to HuggingFace model
            model_source (str): HF class name under which the model is implemented
            training_type: training strategy, peft, finetuning or probing
            tokenizer (AutoTokenizer): tokenize sequences according to the chosen LLM
        """
        super().__init__(
            model_configuration=model_configuration,
            llm_hf_model_path=llm_hf_model_path,
            model_source=model_source,
            training_type=training_type,
            tokenizer=tokenizer,
        )

    def forward(self, **kwargs: Dict) -> torch.Tensor:
        """Definition of the model forward pass.

        Returns:
            torch.Tensor: binary immune response prediction
        """
        x: torch.Tensor = kwargs["pair"]
        attention_mask: torch.Tensor = kwargs["attention_mask"]
        max_length = kwargs["max_length"]

        batch_size = x.shape[0]

        if not attention_mask:
            attention_mask = x != self._tokenizer.cls_token_id

        # find mutated token positions
        mut_token_positions = [find_mut_token(pair[0], pair[1]) for pair in x]

        # batch embeddings of the (wild_type, mutated) sequence pairs
        x = self.llm(
            x.view(batch_size * 2, -1),  # flatten along batch to compute all embeddings in one pass
            attention_mask=attention_mask.view(batch_size * 2, -1),
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )["hidden_states"][-1]

        # group embeddings along batch
        x = torch.reshape(x, (batch_size, 2, max_length, -1))

        # mutation embedding given by the difference between the mutated and wild type
        mutation_embeddings = x[:, 0, :, :] - x[:, 1, :, :]

        # focus on mutated token only
        mutation_embeddings = torch.stack(
            [mutation_embeddings[b, mut_token_positions[b], :] for b in range(batch_size)]
        )

        # pass aggregated embedding through MLP layers
        for layer in self._mlp_layers[:-1]:
            mutation_embeddings = layer(mutation_embeddings)
            mutation_embeddings = self._relu(mutation_embeddings)

        logits = self._mlp_layers[-1](mutation_embeddings)

        return logits


class ProbingModel(TorchBaseModel):
    """Probing Model uses generated embeddings of the mutated and wild type sequences.

    Args:
        Super class (torch.nn.Module): _description_
    """

    def __init__(
        self,
        model_configuration: Dict[str, Any],
    ):
        """Initializes the model object.

        Args:
            model_configuration (Dict[str, Any]): dict containing model configuration
        """
        super().__init__(model_configuration=model_configuration)

    def forward(
        self,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """Definition of the model forward pass.

        Returns:
            torch.Tensor: binary immune response prediction
        """
        x: torch.Tensor = kwargs["pair"]

        # mutation embedding given by the difference between the mutated and wild type
        mutation_embeddings = x[:, 0, :] - x[:, 1, :]

        # pass aggregated embedding through MLP layers
        for layer in self._mlp_layers[:-1]:
            mutation_embeddings = layer(mutation_embeddings)
            mutation_embeddings = self._relu(mutation_embeddings)

        logits = self._mlp_layers[-1](mutation_embeddings)

        return logits
