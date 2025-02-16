import asyncio

import dramatiq
from dramatiq.middleware import Middleware
from redis import Redis

from app._enums import QueueNames, TaskNames, TaskStates
from app.core.config import settings
from app.models.response_models import TaskStateResponse
from app.utils.logger import logger


class TaskTrackingMiddleware(Middleware):
    """
    Middleware to track the execution state of Dramatiq tasks using Redis.

    This middleware provides Redis-based tracking of tasks, storing metadata such as:
    - `task_id`: Unique identifier for the task.
    - `state`: Current execution state (PENDING, PROCESSING, COMPLETED, FAILED).
    - `queue_name`: Name of the queue where the task was enqueued.
    - `task_name`: Name of the task actor.

    It ensures persistence across multiple worker processes.

    Optional enhancements:
    - Expiration time for task states to avoid Redis bloat.
    """

    def __init__(
        self,
        redis_host: str | None = settings.broker_host,
        redis_port: int | None = settings.broker_port,
        redis_db: int = 0,
        task_expiry: int = 3600,
    ) -> None:
        """
        Initializes the middleware with a Redis connection.

        Args:
            redis_host (str): Redis server hostname.
            redis_port (int): Redis server port.
            redis_db (int): Redis database index.
            task_expiry (int): Expiry time for task states in seconds. Defaults to 1 hour.
        """
        if not redis_host or not redis_port:
            raise ValueError("Redis host and port must be provided.")
        self.redis = Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        self.task_expiry = task_expiry
        logger.info("TaskTrackingMiddleware initialized with Redis.")

    def before_enqueue(
        self, broker: dramatiq.Broker, message: dramatiq.Message, delay: int
    ) -> None:
        """
        Called before a task is enqueued. Initializes the task state as 'PENDING'.
        """
        task_id = message.message_id
        self.redis.hset(
            f"task:{task_id}",
            mapping={
                "state": TaskStates.PENDING,
                "queue_name": QueueNames(message.queue_name),
                "task_name": TaskNames(message.actor_name),
            },
        )
        self.redis.expire(f"task:{task_id}", self.task_expiry)
        logger.debug(f"Task {task_id} enqueued and marked as PENDING.")

    def before_process_message(
        self, broker: dramatiq.Broker, message: dramatiq.Message
    ) -> None:
        """
        Called before a task starts processing. Updates state to 'PROCESSING'.
        """
        task_id = message.message_id
        self.redis.hset(f"task:{task_id}", "state", TaskStates.PROCESSING)
        logger.debug(f"Task {task_id} is now PROCESSING.")

    def after_process_message(
        self,
        broker: dramatiq.Broker,
        message: dramatiq.Message,
        *,
        result: None = None,
        exception: Exception | None = None,
    ) -> None:
        """
        Called after a task completes processing. Updates state to 'COMPLETED' or 'FAILED'.
        """
        task_id = message.message_id
        final_state = TaskStates.FAILURE if exception else TaskStates.COMPLETED
        self.redis.hset(f"task:{task_id}", "state", final_state)
        logger.debug(
            f"Task {task_id} has finished processing with state: {final_state}"
        )

    async def get_state(self, task_id: str) -> TaskStateResponse:
        """
        Retrieves the current state of a task by its task_id asynchronously.

        Args:
            task_id (str): The unique identifier of the task message.

        Returns:
            TaskStateResponse: The current state of the task.
        """
        task_data: dict = await asyncio.to_thread(self.redis.hgetall, f"task:{task_id}")  # type: ignore

        if not task_data:
            return TaskStateResponse(task_id=task_id, state=TaskStates.UNKNOWN)

        return TaskStateResponse(
            task_id=task_id,
            state=task_data.get("state", TaskStates.UNKNOWN),
            queue_name=task_data.get("queue_name"),
            task_name=task_data.get("task_name"),
        )
