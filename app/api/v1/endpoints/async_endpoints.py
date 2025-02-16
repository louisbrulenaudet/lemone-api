from fastapi import APIRouter

from app._enums import QueueNames, TaskNames
from app._exceptions import TaskInitalizationError, TaskNotFoundError
from app.models.request_models import (
    ClassificationRequest,
    EmbeddingRequest,
    SimilarityRequest,
)
from app.models.response_models import (
    ClassificationResponse,
    EmbeddingResponse,
    SimilarityResponse,
    TaskResponse,
    TaskStateResponse,
)
from app.services.broker import broker
from app.workers.tasks import (
    classification_task,
    embeddings_task,
    similarity_task,
    task_tracker,
)

router = APIRouter(tags=["async"])


@router.get(
    "/task/status/{task_id}",
    response_model=TaskStateResponse,
    tags=["async", "task", "status"],
    description="Endpoint to get the status of a task using the task ID.",
)
async def get_task_status(task_id: str) -> TaskStateResponse:
    """
    Endpoint to get the status of a task using the task ID

    This endpoint allows clients to get the status of a task by providing the task ID.
    """
    try:
        state: TaskStateResponse | None = await task_tracker.get_state(task_id)

        if isinstance(state, TaskStateResponse):
            return state

    except Exception as exc:
        raise TaskNotFoundError(f"Failed to get task status: {str(exc)}") from exc


@router.post(
    "/embeddings/async",
    response_model=TaskResponse,
    tags=["embeddings", "embedding", "embeddings-vector", "similarity-search"],
    description="Asynchronous endpoint to generate text embeddings for a given input using a registered model.",
)
async def embeddings_async(input_data: EmbeddingRequest) -> TaskResponse:
    """
    Asynchronous endpoint to create embedding generation task

    This endpoint allows clients to send input text along with the model name, and returns the corresponding embeddings generated by the specified model.
    """
    try:
        message = embeddings_task.send(input_data.model_dump())

        return TaskResponse(
            queue_name=QueueNames.EMBEDDING,
            task_name=TaskNames.EMBEDDING,
            task_id=message.message_id,
            task_timestamp=message.message_timestamp,
        )

    except Exception as exc:
        raise TaskInitalizationError(
            f"Failed to create embedding task: {str(exc)}"
        ) from exc


@router.get(
    "/embeddings/async/{task_id}",
    response_model=EmbeddingResponse,
    tags=["embeddings", "embedding", "embeddings-vector", "similarity-search"],
    description="Asynchronous endpoint to get the embeddings generated for a given input using a registered model.",
)
async def get_embeddings_result(
    task_id: str,
) -> EmbeddingResponse:
    """
    Asynchronous endpoint to get embedding generation task result

    This endpoint allows clients to get the result of the embedding generation task by providing the task ID.
    """
    try:
        result = broker.get_result(
            queue_name=QueueNames.EMBEDDING,
            task_name=TaskNames.EMBEDDING,
            task_id=task_id,
        )

        return EmbeddingResponse(**result)

    except Exception as exc:
        raise TaskNotFoundError(
            f"Failed to get embedding task result: {str(exc)}"
        ) from exc


@router.post(
    "/similarity/async",
    response_model=TaskResponse,
    tags=["embeddings", "similarity", "semantic-similarity", "similarity-matrix"],
    description="Endpoint to compute the similarity between two multiple inputs using a registered model.",
)
async def similarity_async(input_data: SimilarityRequest) -> TaskResponse:
    """
    Asynchronous endpoint to create similarity computation task

    This endpoint allows clients to send two input texts along with the model name, and returns the similarity scores matrix computed by the specified model.
    """
    try:
        message = similarity_task.send(input_data.model_dump())

        return TaskResponse(
            queue_name=QueueNames.SIMILARITY,
            task_name=TaskNames.SIMILARITY,
            task_id=message.message_id,
            task_timestamp=message.message_timestamp,
        )

    except Exception as exc:
        raise TaskInitalizationError(
            f"Failed to create similarity task: {str(exc)}"
        ) from exc


@router.get(
    "/similarity/async/{task_id}",
    response_model=SimilarityResponse,
    tags=["embeddings", "similarity", "semantic-similarity", "similarity-matrix"],
    description="Asynchronous endpoint to get the similarity between two multiple inputs using a registered model.",
)
async def get_similarity_result(
    task_id: str,
) -> SimilarityResponse:
    """
    Asynchronous endpoint to get similarity computation task result

    This endpoint allows clients to get the result of the similarity computation task by providing the task ID.
    """
    try:
        result = broker.get_result(
            queue_name=QueueNames.SIMILARITY,
            task_name=TaskNames.SIMILARITY,
            task_id=task_id,
        )

        return SimilarityResponse(**result)

    except Exception as exc:
        raise TaskNotFoundError(
            f"Failed to get similarity task result: {str(exc)}"
        ) from exc


@router.post(
    "/classification/async",
    response_model=TaskResponse,
    tags=["classification", "router", "sequences-classification"],
    description="Endpoint to classify text inputs using a classification model.",
)
async def classification_async(input_data: ClassificationRequest) -> TaskResponse:
    """
    Asynchronous endpoint to create classification computation task

    This endpoint allows clients to classify input text using a registered model.
    """
    try:
        message = classification_task.send(input_data.model_dump())

        return TaskResponse(
            queue_name=QueueNames.CLASSIFICATION,
            task_name=TaskNames.CLASSIFICATION,
            task_id=message.message_id,
            task_timestamp=message.message_timestamp,
        )

    except Exception as exc:
        raise TaskInitalizationError(
            f"Failed to create classification task: {str(exc)}"
        ) from exc


@router.get(
    "/classification/async/{task_id}",
    response_model=ClassificationResponse,
    tags=["classification", "router", "sequences-classification"],
    description="Asynchronous endpoint to get the classification of text inputs using a classification model.",
)
async def get_classification_result(
    task_id: str,
) -> ClassificationResponse:
    """
    Asynchronous endpoint to get classification computation task result

    This endpoint allows clients to get the result of the classification computation task by providing the task ID.
    """
    try:
        result = broker.get_result(
            queue_name=QueueNames.CLASSIFICATION,
            task_name=TaskNames.CLASSIFICATION,
            task_id=task_id,
        )

        return ClassificationResponse(**result)

    except Exception as exc:
        raise TaskNotFoundError(
            f"Failed to get classification task result: {str(exc)}"
        ) from exc
