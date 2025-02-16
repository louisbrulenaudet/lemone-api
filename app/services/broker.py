import time
from collections.abc import Callable
from functools import wraps
from threading import Lock
from typing import Optional, ParamSpec, TypeVar

from dramatiq import Message
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

from app._enums import QueueNames, TaskNames
from app._exceptions import (
    BrokerInitializationError,
    BrokerResultBackendInitializationError,
    TaskNotFoundError,
)
from app.core.config import settings
from app.utils.logger import logger

P = ParamSpec("P")
R = TypeVar("R")

__all__ = ["broker"]


def handle_connection_error(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to handle connection errors and implement retry logic with exponential backoff.

    Retries the decorated function up to 3 times in case of failure, with exponential backoff (1s, 2s, 4s).
    Logs warnings on failures and raises the last encountered exception if all retries fail.

    Args:
        func (Callable[P, R]): The function to be wrapped with retry logic.

    Returns:
        Callable[P, R]: A wrapped function with error handling.

    Example:
        @handle_connection_error
        def connect_to_service():
            # Function implementation that might fail
            pass
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        max_retries = 3
        retry_delay = 1  # seconds
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to execute {func.__name__} after {max_retries} attempts: {str(e)}"
                    )
                    raise last_exception from e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(retry_delay * (2**attempt))  # Exponential backoff

        raise (
            last_exception if last_exception else RuntimeError("Unknown error occurred")
        )

    return wrapper


class Broker:
    """
    Thread-safe singleton class for managing Dramatiq broker and result backend.

    This class ensures that only a single instance of the broker and result backend exists
    throughout the application lifecycle using a singleton pattern with thread safety.
    """

    _instance: Optional["Broker"] = None
    _lock: Lock = Lock()
    _initialized: bool = False

    def __new__(cls) -> "Broker":
        """
        Create a singleton instance of the Broker class.

        Returns:
            Broker: A singleton instance of the broker.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        """
        Initialize the broker manager if it hasn't been initialized.
        Uses a double-checked locking pattern to ensure thread safety.
        """
        with self._lock:
            if not self._initialized:
                self._broker: RedisBroker | None = None
                self._result_backend: RedisBackend | None = None
                self._connection_params = {
                    "host": settings.broker_host,
                    "port": settings.broker_port,
                    "socket_timeout": 5.0,  # Configurable timeout
                    "socket_connect_timeout": 3.0,
                    "retry_on_timeout": True,
                }
                self._initialized = True

    @property
    def broker(self) -> RedisBroker:
        """
        Get the Redis broker instance. Initializes it if not already done.

        Returns:
            RedisBroker: The configured Redis broker instance.
        """
        if self._broker is None:
            self._initialize_broker()

        if self._broker is None:
            raise BrokerInitializationError(
                "Broker instance is not initialized or failed to initialize."
            )

        return self._broker

    @property
    def result_backend(self) -> RedisBackend:
        """
        Get the Redis result backend instance. Initializes it if not already done.

        Returns:
            RedisBackend: The configured Redis result backend instance.
        """
        if self._result_backend is None:
            self._initialize_result_backend()

        if self._result_backend is None:
            raise BrokerResultBackendInitializationError(
                "Result backend instance is not initialized or failed to initialize."
            )

        return self._result_backend

    def get_result(
        self,
        queue_name: QueueNames,
        task_name: TaskNames,
        task_id: str,
        timeout: int = 1000,
    ) -> dict:
        """
        Get the result of a task from the result backend.

        Args:
            queue_name (QueueNames): The name of the queue where the task was sent.
            task_name (TaskNames): The name of the task message.
            task_id (str): The unique identifier of the task message.
            timeout (int, optional): The time in milliseconds to wait for the result. Defaults to 1000.

        Returns:
            dict: The result of the task message.
        """
        try:
            task = Message(
                queue_name=queue_name,
                actor_name=task_name,
                args=(),
                kwargs={},
                options={},
                message_id=task_id,
            )

            return self.result_backend.get_result(
                task,
                timeout=timeout,
            )

        except Exception as exc:
            raise TaskNotFoundError(
                f"Failed to get embedding task result: {str(exc)}"
            ) from exc

    @handle_connection_error
    def _initialize_result_backend(self) -> None:
        """
        Initialize the Redis result backend with error handling.

        Raises:
            BrokerResultBackendInitializationError: If initialization fails after retries.
        """
        try:
            self._result_backend = RedisBackend(**self._connection_params)
            logger.info("Successfully initialized Redis result backend")
        except Exception as exc:
            raise BrokerResultBackendInitializationError(details=exc) from exc

    @handle_connection_error
    def _initialize_broker(self) -> None:
        """
        Initialize the Redis broker with error handling.

        Raises:
            BrokerInitializationError: If initialization fails after retries.
        """
        try:
            self._broker = RedisBroker(**self._connection_params)

            if self._result_backend is None:
                self._initialize_result_backend()

            self._broker.add_middleware(Results(backend=self._result_backend))

            self._broker.client.ping()
            logger.info("Successfully initialized Redis broker")
        except Exception as exc:
            raise BrokerInitializationError(details=exc) from exc


# Singleton instance of the Broker
broker = Broker()
