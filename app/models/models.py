from pydantic import BaseModel, Field

from app._enums import ClassificationLabels, ObjectTypes

__all__ = ["Embedding", "Similarity", "Classification"]


class Embedding(BaseModel):
    input: str = Field(..., description="Original input text")
    index: int = Field(..., description="Index of the input text in the batch")
    object: ObjectTypes = Field(
        default=ObjectTypes.EMBEDDING,
        description="The object type, which is always 'embedding'.",
    )
    embedding: list[float] = Field(
        ...,
        description="Embeddings vectors with flexible constraints which is a list of floats with length depending on the model.",
    )


class Similarity(BaseModel):
    object: ObjectTypes = Field(
        default=ObjectTypes.SIMILARITY,
        description="The object type, which is always 'similarity'.",
    )
    data: list[list[float]] = Field(
        ...,
        description="Similarity matrix between the input texts.",
    )


class Classification(BaseModel):
    object: ObjectTypes = Field(
        default=ObjectTypes.CLASSIFICATION,
        description="The object type, which is always 'classification'.",
    )
    label: ClassificationLabels | str = Field(
        ..., description="The classification label for the input text."
    )
    score: float = Field(
        ..., description="The classification score for the input text."
    )
    index: int = Field(..., description="Index of the input text in the batch")
