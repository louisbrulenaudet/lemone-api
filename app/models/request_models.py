from pydantic import BaseModel, Field

from app._enums import ClassificationModels, EncodingFormats, Models

__all__ = ["EmbeddingRequest", "SimilarityRequest", "ClassificationRequest"]


class EmbeddingRequest(BaseModel):
    input: str | list[str] = Field(
        ...,
        description="Input text to process. Can be either a string or a list of strings.",
    )
    model: Models = Field(
        default=Models.LEMONE_EMBED_PRO, description="Model to use for processing."
    )
    encoding_format: EncodingFormats = Field(
        default=EncodingFormats.FLOAT32,
        description="Encoding format for the response, can be `float32`, `int8`, `uint8`, or `binary`. Defaults to `float32`",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "input": "This is a sample input to the API.",
                "model": "lemone-embed-pro",
                "encoding_format": "float32",
            }
        }


class SimilarityRequest(BaseModel):
    input: list[str] = Field(
        ..., description="Input texts to process. Must be a list of strings."
    )
    model: Models = Field(
        default=Models.LEMONE_EMBED_PRO, description="Model to use for processing."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "input": [
                    "This is a sample input to the API.",
                    "This is another input.",
                ],
                "model": "lemone-embed-pro",
            }
        }


class ClassificationRequest(BaseModel):
    input: str | list[str] = Field(
        ...,
        description="Input text to process. Can be either a string or a list of strings.",
    )
    model: ClassificationModels = Field(
        default=ClassificationModels.LEMONE_ROUTER_L,
        description="Model to use for classification",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "input": "This is a sample input to classify.",
                "model": "lemone-router-l",
            }
        }
