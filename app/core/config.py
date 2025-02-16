from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic.fields import Field
from pydantic_settings import BaseSettings
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available

from app._enums import ClassificationModels, DeviceTypes, EmbeddingBackend, Models
from app.core.classification import ClassificationModel
from app.core.registries import ModelRegistry


class Settings(BaseSettings):
    """
    Configuration settings for the application, using Pydantic for validation.
    """

    name: str = Field(default="Embeddings API", alias="APP_NAME")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    logfire_token: str | None = Field(default=None, alias="LOGFIRE_TOKEN")
    broker_port: int | None = Field(default=None, alias="BROKER_PORT")
    broker_host: str | None = Field(default=None, alias="BROKER_HOST")
    worker_threads: int = Field(default=4, alias="WORKER_THREADS")
    default_device: DeviceTypes = Field(default=DeviceTypes.CPU, alias="DEVICE")
    model_names: list[Models | ClassificationModels] = [
        Models.LEMONE_EMBED_PRO,
        ClassificationModels.LEMONE_ROUTER_L,
    ]
    device: DeviceTypes = DeviceTypes.CPU
    backend: EmbeddingBackend = EmbeddingBackend.TORCH
    registry: ModelRegistry | None = None
    tokenizers_parallelism: bool = Field(default=True, alias="TOKENIZERS_PARALLELISM")

    @model_validator(mode="before")
    @classmethod
    def detect_device(cls, values: dict) -> dict:
        """
        Automatically detect the best available device for model operations.

        This method checks for the availability of CUDA (GPU) and MPS (Apple Silicon) devices.
        If neither is available, it defaults to CPU.

        Args:
            values (dict): Initial settings values provided by Pydantic.

        Returns:
            dict: Updated settings values with the detected device.

        Example:
            If a GPU is available:
            ```
            values["device"] = DeviceTypes.CUDA
            ```
        """
        if values.get("default_device") is not None:
            values["device"] = values["default_device"]
        else:
            values["device"] = next(
                (
                    device
                    for device, check in {
                        DeviceTypes.CUDA: is_cuda_available,
                        DeviceTypes.MPS: is_mps_available,
                    }.items()
                    if check()
                ),
                DeviceTypes.CPU,  # Fallback to CPU if no other device is available
            )
        return values

    @classmethod
    @lru_cache
    def map_model(cls, model_name: str) -> Models:
        """
        Map a model name (string) to a `Models` enumeration instance.

        This function provides a mechanism to validate and translate string-based model names into their corresponding enumeration values, ensuring compatibility.

        Args:
            model_name (str): The name of the model to map.

        Returns:
            Models: The corresponding instance of the `Models` enumeration.

        Raises:
            ValueError: If the provided model name is not valid.

        Example:
            ```
            Settings.map_model("LEMONE_EMBED_PRO")  # Returns Models.LEMONE_EMBED_PRO
            ```
        """
        try:
            return Models[model_name]

        except KeyError as exc:
            raise ValueError(
                f"Invalid model name '{model_name}'. Available models: {list(Models)}"
            ) from exc

    def load_models(self) -> ModelRegistry:
        """
        Load and register models specified in the settings, initializing the model registry.

        This function dynamically creates an instance of the `ModelRegistry`, registers the models defined in the `model_names` attribute, and stores the registry for use throughout the application.

        Returns:
            ModelRegistry: The initialized registry containing the loaded models.

        Example:
            ```
            registry = settings.load_models()
            registry.get_model("LEMONE_EMBED_PRO")
            ```
        """
        from app.core.classification import ClassificationModel
        from app.core.embeddings import EmbeddingModel

        def model_factory(model_name: str) -> EmbeddingModel | ClassificationModel:
            """
            Factory function to instantiate different model types based on the model name.

            This function is used by the `ModelRegistry` to create instances of the appropriate model type based on the model name provided.

            Args:
                model_name (str): The name of the model to instantiate.

            Returns:
                EmbeddingModel | ClassificationModel: An instance of the corresponding model type.
            """
            if model_name in Models:
                return EmbeddingModel(
                    model_name, backend=self.backend, device=self.device
                )

            elif model_name in ClassificationModels:
                return ClassificationModel(model_name, device=self.device)

            else:
                raise ValueError(f"Invalid model name '{model_name}'.")

        self.registry = ModelRegistry(model_factory=model_factory)

        self.registry.register_models(
            model_names=self.model_names,
        )

        return self.registry

    class Config:
        """
        Configuration for the Pydantic `BaseSettings` class.

        This class specifies that environment variables should be loaded from a `.env` file.
        """

        env_file = ".env"


# Initialize the settings object globally
settings = Settings()  # type: ignore
settings.load_models()
