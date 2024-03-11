"""Definition of the LLMBasedModel class."""
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer


class BaseModel(torch.nn.Module):
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
            attention_mask (Optional[np.array], optional): attention mask
        """
        super().__init__()
        self._model_configuration = model_configuration
        self._llm_hf_model_path = model_configuration["llm_hf_model_path"]
        self._llm = AutoModelForMaskedLM.from_pretrained(
            self._llm_hf_model_path, trust_remote_code=True
        )

        mlp_layer_sizes = self._model_configuration["mlp_layer_sizes"]
        if not mlp_layer_sizes:
            mlp_layer_sizes = [self._model_configuration["embed_dim"]]

        mlp_layer_sizes += [1]

        self._mlp_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(mlp_layer_sizes[idx], mlp_layer_sizes[idx + 1])
                for idx in range(len(mlp_layer_sizes) - 1)
            ]
        )
        self._relu = torch.nn.ReLU()
        self._sigmoid = torch.nn.Sigmoid()


class FinetuningModel(BaseModel):
    """Classifies the aggregated embeddings of the mutated and wild type sequences.

    Args:
        Super class (torch.nn.Module): _description_
    """

    def __init__(
        self,
        model_configuration: Dict[str, Any],
        model_type: str,
        tokenizer: AutoTokenizer,
    ):
        """Initializes the model object.

        Args:
            model_configuration (Dict[str, Any]): dict containing model configuration
            model_type (str): finetuning_model or peft_model
            tokenizer (AutoTokenizer): tokenize sequences according to the chosen LLM
        """
        super().__init__(model_configuration)
        self._tokenizer = tokenizer
        if model_type == "peft_model":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "key", "value"],
            )
            self._llm = get_peft_model(self._llm, peft_config)

    def forward(
        self,
        x: torch.Tensor,
        max_length: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Definition of the model forward pass.

        Returns:
            torch.Tensor: binary immune response prediction
        """
        batch_size = x.shape[0]

        if not attention_mask:
            attention_mask = x != self._tokenizer.cls_token_id

        # batch embeddings of the (wild_type, mutated) sequence pairs
        x = self._llm(
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
