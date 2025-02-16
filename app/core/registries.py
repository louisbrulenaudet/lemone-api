from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Generic, TypeVar

from app._enums import ClassificationModels, Models

ModelType = TypeVar("ModelType")


class ModelRegistry(Generic[ModelType]):
    """
    A thread-safe base registry for managing and storing instances of machine learning models.

    This class provides a generic implementation for model registration, retrieval, and listing functionality. It ensures thread safety for concurrent access and utilizes a factory method to dynamically create model instances when needed.

    The class is designed to be inherited by specific model registry implementations, such as embedding or classification model registries.

    Type Parameters:
        ModelType: The type of model this registry will manage (e.g., EmbeddingModel, ClassificationModel)
    """

    def __init__(self, model_factory: Callable[[str], ModelType]) -> None:
        """
        Initialize the BaseRegistry.

        Args:
            model_factory (Callable[[str], ModelType]):
                A callable function or method that accepts a model name (str)
                and returns an instance of the corresponding model type.
        """
        self._models: dict[str, ModelType] = {}
        self._lock = threading.Lock()  # Ensures thread-safe operations
        self._model_factory = model_factory

    def register_model(self, model_name: str, alias: str | None = None) -> ModelType:
        """
        Register a new model in the registry.

        If an alias is provided, it will be used as the key to reference the model;
        otherwise, the model name itself will be used as the key.

        Args:
            model_name (str):
                The name of the model to register.
            alias (str | None, optional):
                An optional alias to reference the model. Defaults to None.

        Returns:
            ModelType: The newly registered model instance.

        Raises:
            KeyError: If a model with the given alias or name is already registered.
        """
        model_key = alias or model_name

        with self._lock:
            if model_key in self._models:
                raise KeyError(f"Model '{model_key}' is already registered.")

            model = self._model_factory(model_name)
            self._models[model_key] = model

        return model

    def register_models(self, model_names: list[Models | ClassificationModels]) -> None:
        """
        Register multiple models at once.

        Args:
            model_names (list[Models | ClassificationModels]):
                A list of model names, Models enum or ClassificationModels enum values.

        Returns:
            None
        """
        for model_name in model_names:
            self.register_model(model_name)

    def get_model(self, model_key: str) -> ModelType:
        """
        Retrieve a registered model by its key (name or alias).

        Args:
            model_key (str):
                The key used to reference the model (either its name or alias).

        Returns:
            ModelType: The model instance corresponding to the given key.

        Raises:
            KeyError: If the model key is not found in the registry.
        """
        with self._lock:
            if model_key not in self._models:
                raise KeyError(f"Model '{model_key}' not found in the registry.")

            return self._models[model_key]

    def list_registered_models(self) -> list[str]:
        """
        List all registered models' keys (names and aliases).

        Returns:
            list[str]: A list of keys (model names or aliases) for all registered models.
        """
        with self._lock:
            return list(self._models.keys())
