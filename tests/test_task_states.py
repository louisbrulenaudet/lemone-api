import pytest

from app._enums import QueueNames, TaskNames, TaskStates
from app._exceptions import TaskNotFoundError, TaskTrackingError
from app.models.response_models import TaskStateResponse


def test_task_state_response_model():
    """Test TaskStateResponse model creation and validation."""
    state_response = TaskStateResponse(
        task_id="test-task-id",
        state=TaskStates.COMPLETED,
        queue_name=QueueNames.EMBEDDING,
        task_name=TaskNames.EMBEDDING,
    )
    assert state_response.task_id == "test-task-id"
    assert state_response.state == TaskStates.COMPLETED
    assert state_response.queue_name == QueueNames.EMBEDDING
    assert state_response.task_name == TaskNames.EMBEDDING


def test_task_state_response_pending():
    """Test TaskStateResponse with pending state."""
    state_response = TaskStateResponse(
        task_id="pending-task",
        state=TaskStates.PENDING,
        queue_name=QueueNames.CLASSIFICATION,
        task_name=TaskNames.CLASSIFICATION,
    )
    assert state_response.state == TaskStates.PENDING


def test_task_state_response_processing():
    """Test TaskStateResponse with processing state."""
    state_response = TaskStateResponse(
        task_id="processing-task",
        state=TaskStates.PROCESSING,
        queue_name=QueueNames.SIMILARITY,
        task_name=TaskNames.SIMILARITY,
    )
    assert state_response.state == TaskStates.PROCESSING


def test_task_state_response_failure():
    """Test TaskStateResponse with failure state."""
    state_response = TaskStateResponse(
        task_id="failed-task",
        state=TaskStates.FAILURE,
        queue_name=QueueNames.EMBEDDING,
        task_name=TaskNames.EMBEDDING,
    )
    assert state_response.state == TaskStates.FAILURE


def test_task_state_response_unknown():
    """Test TaskStateResponse with unknown state."""
    state_response = TaskStateResponse(
        task_id="unknown-task",
        state=TaskStates.UNKNOWN,
        queue_name=QueueNames.EMBEDDING,
        task_name=TaskNames.EMBEDDING,
    )
    assert state_response.state == TaskStates.UNKNOWN


def test_task_state_raises_not_found():
    """Test that attempting to get a non-existent task raises TaskNotFoundError."""
    with pytest.raises(TaskNotFoundError) as exc_info:
        raise TaskNotFoundError("non-existent-task")
    assert "non-existent-task" in str(exc_info.value)
    assert exc_info.value.code == "TASK_NOT_FOUND"


def test_task_state_raises_tracking_error():
    """Test that task tracking issues raise TaskTrackingError."""
    with pytest.raises(TaskTrackingError) as exc_info:
        raise TaskTrackingError("Failed to track task")
    assert "Failed to track task" in str(exc_info.value)
    assert exc_info.value.code == "TASK_TRACKING_ERROR"
