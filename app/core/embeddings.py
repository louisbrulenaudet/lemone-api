from __future__ import annotations

from collections.abc import Generator

from sentence_transformers import SentenceTransformer

from app._enums import DeviceTypes, EmbeddingBackend, SimilarityFunctions
from app._exceptions import (
    EmbeddingComputeError,
    ModelNotFoundError,
    SimilarityComputeError,
)
from app.models.models import Embedding, Similarity

__all__ = ["EmbeddingModel"]


class EmbeddingModel:
    """
    A class for managing and using SentenceTransformer models to generate text embeddings.

    This class provides methods to load a model, manage its configuration, and encode text inputs into embeddings. It supports options for normalizing the embeddings and is designed for extensibility and ease of use.
    """

    def __init__(
        self,
        model_name: str,
        backend: EmbeddingBackend = EmbeddingBackend.TORCH,
        device: DeviceTypes = DeviceTypes.CPU,
    ) -> None:
        """
        Initialize an EmbeddingModel instance with the specified model name.

        Args:
            model_name (str):
                The name or identifier of the SentenceTransformer model to load.
            backend (EmbeddingBackend):
                The backend to use for encoding text inputs, depends on if the execution machine has a GPU or not, if a reasonable loss in accuracy is acceptable, and the brand of the CPU.
            device (DeviceTypes, optional):
                The device type to use for encoding text inputs. Defaults to DeviceTypes.CPU.

        Attributes:
            model_name (str): Name of the loaded SentenceTransformer model.
            device (DeviceTypes): The device type to use for encoding text inputs.
            backend (EmbeddingBackend): The backend to use for encoding text inputs.
            _model (SentenceTransformer): Loaded SentenceTransformer model instance.
        """
        self.model_name = model_name
        self.device: DeviceTypes = device
        self.backend: EmbeddingBackend = backend
        self._model: SentenceTransformer = self.load_model()
        self._model.similarity_fn_name = SimilarityFunctions.COSINE

    def load_model(self) -> SentenceTransformer:
        """
        Load the SentenceTransformer model for encoding text.

        The model is loaded onto the specified device (e.g., CPU, GPU) and
        uses the `trust_remote_code` flag for remote repositories.

        Returns:
            SentenceTransformer: An instance of the loaded SentenceTransformer model.

        Raises:
            ModelNotFoundError: If the specified model is not found on the remote repository or local cache.
        """
        try:
            return SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )

        except Exception as exc:
            raise ModelNotFoundError(model_name=self.model_name, details=exc) from exc

    def iter_encode(
        self, sentences: list[str] | str, normalize_embeddings: bool = True
    ) -> Generator[Embedding, None, None]:
        """
        Encode a list of sentences into embeddings using the loaded model.

        Args:
            sentences (list[str]):
                A list of input sentences or phrases to encode.
            normalize_embeddings (bool, optional):
                Whether to normalize the resulting embeddings. Defaults to True.

        Yields:
            Embedding: An instance of the Embedding model containing:
                - `input` (str): The original input sentence.
                - `embedding` (list[float]): The embedding vector for the sentence.
                - `index` (int): The index of the sentence in the input list.

        Raises:
            Exception: If the model instance is not loaded or fails during encoding.

        Example:
            >>> model = EmbeddingModel("louisbrulenaudet/lemone-embed-pro")
            >>> sentences = ["Hello, world!", "How are you?"]
            >>> for embedding in model.encode(sentences):
            >>>     print(embedding.input, embedding.embedding, embedding.index)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        if self._model is None:
            self.load_model()

        try:
            embeddings = self._model.encode(
                sentences, normalize_embeddings=normalize_embeddings
            ).tolist()

            for i, embedding in enumerate(embeddings):
                yield Embedding(
                    input=sentences[i],
                    embedding=embedding,  # type: ignore for type hints
                    index=i,
                )

        except Exception as exc:
            raise EmbeddingComputeError(details=exc) from exc

    def encode(
        self, sentences: list[str] | str, normalize_embeddings: bool = True
    ) -> list[Embedding]:
        """
        Encode a list of sentences into embeddings using the loaded model.

        Args:
            sentences (list[str]):
                A list of input sentences or phrases to encode.
            normalize_embeddings (bool, optional):
                Whether to normalize the resulting embeddings. Defaults to True.

        Returns:
            list[Embedding]: A list of Embedding instances containing:
                - `input` (str): The original input sentence.
                - `embedding` (list[float]): The embedding vector for the sentence.
                - `index` (int): The index of the sentence in the input list.

        Raises:
            Exception: If the model instance is not loaded or fails during encoding.

        Example:
            >>> model = EmbeddingModel("louisbrulenaudet/lemone-embed-pro")
            >>> sentences = ["Hello, world!", "How are you?"]
            >>> embeddings = model.encode(sentences)
            >>> for embedding in embeddings:
            >>>     print(embedding.input, embedding.embedding, embedding.index)
            ... "Hello, world!", [0.1, 0.2, ...], 0
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        return list(self.iter_encode(sentences, normalize_embeddings))

    def similarity(
        self, sentences: list[str], normalize_embeddings: bool = True
    ) -> Similarity:
        """
        Compute the similarity matrix between a list of sentences using the loaded model.

        Args:
            sentences (list[str]):
                A list of input sentences or phrases to compute similarities.
            normalize_embeddings (bool, optional):
                Whether to normalize the resulting embeddings. Defaults to True.

        Returns:
            Similarity: An instance of the Similarity model containing:
                - `data` (list[list[float]]): The similarity matrix between the input texts.

        Raises:
            EmbeddingComputeError: If the model fails during encoding.
            SimilarityComputeError: If the model fails during similarity computation.

        Example:
            >>> model = EmbeddingModel("louisbrulenaudet/lemone-embed-pro")
            >>> sentences = ["Hello, world!", "How are you?"]
            >>> similarity = model.similarity(sentences)
            >>> print(similarity.data)
            ... [[1.0, 0.5], [0.5, 1.0]]
        """
        try:
            embeddings = self._model.encode(
                sentences, normalize_embeddings=normalize_embeddings
            )

        except Exception as exc:
            raise EmbeddingComputeError(details=exc) from exc

        try:
            similarities = self._model.similarity(
                embeddings,
                embeddings,
            )

        except Exception as exc:
            raise SimilarityComputeError(details=exc) from exc

        return Similarity(data=similarities.tolist())
