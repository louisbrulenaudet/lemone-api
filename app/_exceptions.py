from __future__ import annotations

from typing import Any

from app._enums import ErrorCodes
from app.utils.logger import logger

__all__ = [
    "CoreError",
    "BrokerInitializationError",
    "BrokerResultBackendInitializationError",
    "ClassificationComputeError",
    "EmbeddingComputeError",
    "ModelNotFoundError",
    "ModelRegistryNotFoundError",
    "SimilarityComputeError",
    "TaskInitalizationError",
    "TaskNotFoundError",
    "TaskTrackingError",
]


class CoreError(Exception):
    """
    A custom exception class for handling application-specific errors.

    This exception includes an error message, an error code, and optional details.
    It also logs the error upon initialization.

    Attributes:
        message (str): A descriptive error message.
        code (ErrorCodes): An enumerated error code representing the specific error type.
        details (dict[str, Any] | str | None): Additional details about the error.
    """

    message: str
    code: ErrorCodes
    details: dict[str, Any] | str | None

    def __init__(
        self,
        message: str,
        code: ErrorCodes,
        details: dict[str, Any] | str | None = None,
    ) -> None:
        """
        Initialize a CoreError instance with an error message, code, and optional details.

        The error is logged automatically when an instance is created.

        Args:
            message (str): The error message.
            code (ErrorCodes): A predefined error code representing the error type.
            details (dict[str, Any] | str | None, optional): Additional information about the error.
                Can be a dictionary, a string, or None. Defaults to None.
        """
        self.message = message
        self.code = code
        self.details = details

        logger.error(
            f"{self.__class__.__name__}: {message} [Code: {code}] Details: {details}"
        )

    def __str__(self) -> str:
        """
        Return a string representation of the error, including the message, code, and optional details.

        Returns:
            str: A formatted string describing the error.

        Example:
            >>> error = CoreError("Invalid input", ErrorCodes.INVALID_INPUT, {"field": "email"})
            >>> print(str(error))
            "CoreError: Invalid input [Code: INVALID_INPUT] Details: {'field': 'email'}"
        """
        detail_part = f" Details: {self.details}" if self.details else ""
        return f"{self.__class__.__name__}: {self.message} [Code: {self.code}]{detail_part}"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the CoreError instance into a dictionary format.

        This is useful for structured logging or returning errors in API responses.

        Returns:
            dict[str, Any]: A dictionary containing error details.

        Example:
            >>> error = CoreError("Access denied", ErrorCodes.PERMISSION_DENIED, "User lacks admin rights")
            >>> error.to_dict()
            {
                "error": "CoreError",
                "message": "Access denied",
                "code": "PERMISSION_DENIED",
                "details": "User lacks admin rights"
            }
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details or {},
        }


class BrokerInitializationError(CoreError):
    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "The broker initialization failed.",
            ErrorCodes.BROKER_INITIALIZATION_ERROR,
            details=str(details),
        )


class BrokerResultBackendInitializationError(CoreError):
    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "The broker result backend initialization failed.",
            ErrorCodes.BROKER_RESULT_BACKEND_INITIALIZATION_ERROR,
            details=str(details),
        )


class ClassificationComputeError(CoreError):
    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "The classification computation failed.",
            ErrorCodes.CLASSIFICATION_COMPUTE_ERROR,
            details=str(details),
        )


class ModelRegistryNotFoundError(CoreError):
    def __init__(self) -> None:
        super().__init__(
            "The embedding registry was not found.",
            ErrorCodes.MODEL_REGISTRY_NOT_FOUND,
            "This error occurs when the embedding registry is not available or has not been initialized.",
        )


class EmbeddingComputeError(CoreError):
    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "The embedding computation failed.",
            ErrorCodes.EMBEDDING_COMPUTE_ERROR,
            details=str(details),
        )


class ModelNotFoundError(CoreError):
    def __init__(self, model_name: str, details: Exception | str) -> None:
        super().__init__(
            f"The model '{model_name}' was not found.",
            ErrorCodes.MODEL_NOT_EXIST,
            f"Ensure that the model is registered in the embedding registry. Details: {details}",
        )


class SimilarityComputeError(CoreError):
    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "The similarity computation failed.",
            ErrorCodes.SIMILARITY_COMPUTE_ERROR,
            details=str(details),
        )


class TaskInitalizationError(CoreError):
    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "The task initialization failed.",
            ErrorCodes.TASK_INITIALIZATION_ERROR,
            details=str(details),
        )


class TaskNotFoundError(CoreError):
    def __init__(self, task_id: str) -> None:
        super().__init__(
            f"The task with ID '{task_id}' was not found.",
            ErrorCodes.TASK_NOT_FOUND,
            "Ensure that the task ID is correct and the task has not expired.",
        )


class TaskTrackingError(CoreError):
    def __init__(self, details: Exception | str) -> None:
        super().__init__(
            "The task tracking failed.",
            ErrorCodes.TASK_TRACKING_ERROR,
            details=str(details),
        )
