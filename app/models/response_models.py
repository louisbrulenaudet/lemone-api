from pydantic import BaseModel, Field

from app._enums import (
    ClassificationModels,
    Models,
    ObjectTypes,
    QueueNames,
    TaskNames,
    TaskStates,
)
from app.models.models import Classification, Embedding, Similarity

__all__ = [
    "EmbeddingResponse",
    "SimilarityResponse",
    "ClassificationResponse",
    "TaskResponse",
    "TaskStateResponse",
]


class EmbeddingResponse(BaseModel):
    model: Models | str = Field(
        Models.LEMONE_EMBED_PRO, description="Model used for processing"
    )
    object: ObjectTypes = Field(
        default=ObjectTypes.LIST,
        description="The object type, which is always 'list'",
    )
    data: list[Embedding] | Embedding = Field(
        ...,
        description="Embeddings vectors with flexible constraints which is a list of floats with length depending on the model",
    )


class SimilarityResponse(BaseModel):
    model: Models = Field(
        default=Models.LEMONE_EMBED_PRO, description="Model used for processing"
    )
    object: ObjectTypes = Field(
        default=ObjectTypes.LIST,
        description="The object type, which is always 'list'",
    )
    data: Similarity = Field(
        ...,
        description="Similarity matrix between the input texts",
    )


class ClassificationResponse(BaseModel):
    model: ClassificationModels = Field(
        default=ClassificationModels.LEMONE_ROUTER_L,
        description="Model used for processing",
    )
    object: ObjectTypes = Field(
        default=ObjectTypes.LIST,
        description="The object type, which is always 'list'.",
    )
    data: list[Classification] = Field(
        ...,
        description="Classification labels for the input texts",
    )


class TaskResponse(BaseModel):
    queue_name: QueueNames = Field(
        ...,
        description="The name of the queue where the task was sent",
    )
    task_name: TaskNames = Field(
        ...,
        description="The name of the task message",
    )
    task_id: str = Field(
        ...,
        description="The unique identifier of the task message",
    )
    task_timestamp: int = Field(
        ...,
        description="The timestamp when the task was created",
    )


class TaskStateResponse(BaseModel):
    task_id: str = Field(..., description="The unique identifier of the task message")
    state: TaskStates | None = Field(
        default=None, description="The state of the task message"
    )
    queue_name: QueueNames | None = Field(
        default=None, description="The name of the queue where the task was sent"
    )
    task_name: TaskNames | None = Field(
        default=None, description="The name of the task message"
    )
