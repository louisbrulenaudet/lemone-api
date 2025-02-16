from typing import Any

import dramatiq

from app._enums import QueueNames, TaskNames, WorkerMaxRetries
from app._exceptions import (
    ClassificationComputeError,
    EmbeddingComputeError,
    ModelRegistryNotFoundError,
    SimilarityComputeError,
)
from app.core.config import settings
from app.models.response_models import (
    ClassificationResponse,
    EmbeddingResponse,
    SimilarityResponse,
)
from app.services.broker import broker
from app.services.middlewares import TaskTrackingMiddleware

task_tracker = TaskTrackingMiddleware()

dramatiq.set_broker(broker.broker)
dramatiq.get_broker().add_middleware(task_tracker)


@dramatiq.actor(
    queue_name=QueueNames.EMBEDDING,
    actor_name=TaskNames.EMBEDDING,
    max_retries=WorkerMaxRetries.EMBEDDING,
    store_results=True,
)
def embeddings_task(input_data: dict[str, str]) -> dict[str, Any]:
    """
    Endpoint to generate text embeddings for a given input using a registered model.

    This endpoint allows clients to send input text along with the model name, and returns the corresponding embeddings generated by the specified model.
    """
    import time

    time.sleep(10)
    if settings.registry is not None:
        try:
            model = input_data.get("model")
            input = input_data.get("input")

            if model and input:
                data = settings.registry.get_model(model).encode(input)
                return EmbeddingResponse(model=model, data=data).model_dump()

            else:
                raise ValueError("Invalid input data provided.")

        except Exception as exc:
            raise EmbeddingComputeError(details=exc) from exc

    else:
        raise ModelRegistryNotFoundError()


@dramatiq.actor(
    queue_name=QueueNames.SIMILARITY,
    actor_name=TaskNames.SIMILARITY,
    max_retries=WorkerMaxRetries.SIMILARITY,
    store_results=True,
)
def similarity_task(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Endpoint to compute the similarity between two multiple inputs using a registered model.
    """
    if settings.registry is not None:
        try:
            model = input_data.get("model")
            input = input_data.get("input")

            if model and input:
                data = settings.registry.get_model(model).similarity(input)
                return SimilarityResponse(model=model, data=data).model_dump()

            else:
                raise ValueError("Invalid input data provided.")

        except Exception as exc:
            raise SimilarityComputeError(details=exc) from exc

    else:
        raise ModelRegistryNotFoundError()


@dramatiq.actor(
    queue_name=QueueNames.CLASSIFICATION,
    actor_name=TaskNames.CLASSIFICATION,
    max_retries=WorkerMaxRetries.CLASSIFICATION,
    store_results=True,
)
def classification_task(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Endpoint to classify input text using a registered model.
    """
    if settings.registry is not None:
        try:
            model = input_data.get("model")
            input = input_data.get("input")

            if model and input:
                data = settings.registry.get_model(model).classify(input)
                return ClassificationResponse(model=model, data=data).model_dump()

            else:
                raise ValueError("Invalid input data provided.")

        except Exception as exc:
            raise ClassificationComputeError(details=exc) from exc

    else:
        raise ModelRegistryNotFoundError()
