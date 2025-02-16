from __future__ import annotations

from collections.abc import Generator

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    pipeline,
)

from app._enums import DeviceTypes
from app._exceptions import (
    ClassificationComputeError,
    ModelNotFoundError,
)
from app.models.models import Classification

__all__ = ["ClassificationModel"]


class ClassificationModel:
    """
    A class for managing and using Hugging Face Transformer models to classify text.

    This class supports:
    - Loading pretrained models and tokenizers.
    - Using a classification pipeline to process text inputs.
    - Returning structured classification results, with confidence scores and labels.
    """

    def __init__(self, model_name: str, device: DeviceTypes = DeviceTypes.CPU) -> None:
        """
        Initialize a ClassificationModel instance with the specified model.

        Args:
            model_name (str):
                The name or identifier of the Hugging Face model to load.

        Attributes:
            model_name (str): Name of the loaded Hugging Face model.
            _model (AutoModelForSequenceClassification): Loaded Hugging Face classification model.
            _tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer for processing text inputs.
            _pipeline (pipeline): Hugging Face text classification pipeline.
        """
        self.model_name = model_name
        self.device: DeviceTypes = device
        self._model: AutoModelForSequenceClassification = self.load_model()
        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.load_tokenizer()
        )
        self._pipeline = pipeline(
            "text-classification",
            model=self._model,  # type: ignore
            tokenizer=self._tokenizer,
        )

    def load_model(self) -> AutoModelForSequenceClassification:
        """
        Load the pretrained classification model.

        Returns:
            AutoModelForSequenceClassification: An instance of the loaded model.

        Raises:
            ModelNotFoundError: If the model cannot be loaded from the Hugging Face model hub or local cache.
        """
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name, device_map=self.device
            )

        except Exception as exc:
            raise ModelNotFoundError(
                model_name=self.model_name,
                details=f"Failed to load classification model: {exc}",
            ) from exc

    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        Load the tokenizer associated with the specified classification model.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The loaded tokenizer instance.

        Raises:
            ModelNotFoundError: If the tokenizer cannot be loaded.
        """
        try:
            return AutoTokenizer.from_pretrained(self.model_name)

        except Exception as exc:
            raise ModelNotFoundError(
                model_name=self.model_name,
                details=f"Failed to load tokenizer for classification model: {exc}",
            ) from exc

    def iter_classify(
        self, texts: list[str] | str
    ) -> Generator[Classification, None, None]:
        """
        Classify input text(s) and return the results as a generator.

        Args:
            texts (list[str] | str): A single string or a list of strings to classify.

        Yields:
            Classification: An instance containing:
                - `label` (str): The predicted class label.
                - `score` (float): The confidence score of the prediction.
                - `index` (int): The index of the text in the input list.

        Raises:
            ClassificationComputeError: If text classification fails.

        Example:
            >>> model = ClassificationModel("distilbert-base-uncased")
            >>> texts = ["input": "Pour les cessions d’actions, de parts de fondateurs ou de parts de bénéficiaires de sociétés par action, autres que celles des personnes morales à prépondérance immobilière, ainsi que pour les parts ou titres de capital souscrits par les clients des établissements mutualistes ou coopératifs, le droit d’enregistrement est fixé à 0,1 %."]
            >>> for result in model.iter_classify(texts):
            >>>     print(result.label, result.score, result.index)
            ...     "Patrimoine et enregistrement", 0.9997046589851379, 0
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            predictions = self._pipeline(texts)

            for i, prediction in enumerate(predictions):  # type: ignore
                yield Classification(
                    **prediction,  # type: ignore
                    index=i,
                )

        except Exception as exc:
            raise ClassificationComputeError(
                details=f"Failed to classify text: {exc}"
            ) from exc

    def classify(self, text: list[str] | str) -> list[Classification]:
        """
        Classify input text(s) and return the results as a list.

        Args:
            text (list[str] | str): A single string or a list of strings to classify.

        Returns:
            list[Classification]: A list of Classification instances, each containing:
                - `label` (str): The predicted class label.
                - `score` (float): The confidence score of the prediction.
                - `index` (int): The index of the text in the input list.

        Raises:
            ClassificationComputeError: If text classification fails.

        Example:
            >>> model = ClassificationModel("distilbert-base-uncased")
            >>> texts = ["input": "Pour les cessions d’actions, de parts de fondateurs ou de parts de bénéficiaires de sociétés par action, autres que celles des personnes morales à prépondérance immobilière, ainsi que pour les parts ou titres de capital souscrits par les clients des établissements mutualistes ou coopératifs, le droit d’enregistrement est fixé à 0,1 %."]
            >>> for result in model.classify(texts):
            >>>     print(result.label, result.score, result.index)
            ...     "Patrimoine et enregistrement", 0.9997046589851379, 0
        """
        if isinstance(text, str):
            text = [text]

        return list(self.iter_classify(text))
