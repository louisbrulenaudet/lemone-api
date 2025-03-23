from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app._enums import (
    ClassificationModels,
    Models,
    ObjectTypes,
    QueueNames,
    TaskNames,
    TaskStates,
)
from app._exceptions import TaskNotFoundError
from app.models.response_models import TaskStateResponse


def test_task_status_success(client: TestClient, monkeypatch):
    """Test successful task status retrieval."""
    # Mock task state
    mock_state = TaskStateResponse(
        task_id="test-task-id",
        state=TaskStates.COMPLETED,
        queue_name=QueueNames.EMBEDDING,
        task_name=TaskNames.EMBEDDING
    )

    # Mock task_tracker.get_state
    async def mock_get_state(task_id):
        assert task_id == "test-task-id"
        return mock_state

    from app.workers.tasks import task_tracker
    monkeypatch.setattr(task_tracker, "get_state", mock_get_state)

    response = client.get("/api/v1/task/status/test-task-id")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id"
    assert data["state"] == TaskStates.COMPLETED
    assert data["queue_name"] == QueueNames.EMBEDDING
    assert data["task_name"] == TaskNames.EMBEDDING


def test_task_status_not_found(client: TestClient, monkeypatch):
    """Test task status retrieval for non-existent task."""
    async def mock_get_state(task_id):
        raise TaskNotFoundError(f"Task with ID '{task_id}' not found")

    from app.workers.tasks import task_tracker
    monkeypatch.setattr(task_tracker, "get_state", mock_get_state)

    response = client.get("/api/v1/task/status/non-existent-id")
    assert response.status_code == 404
    assert response.json()["error"] == "TaskNotFoundError"


def test_embeddings_async_creation(client: TestClient, monkeypatch):
    """Test successful async embeddings task creation."""
    # Mock message with required attributes
    mock_message = MagicMock()
    mock_message.message_id = "test-message-id"
    mock_message.message_timestamp = 1234567890

    from app.workers.tasks import embeddings_task
    monkeypatch.setattr(embeddings_task, "send", lambda x: mock_message)

    response = client.post(
        "/api/v1/embeddings/async",
        json={"model": Models.LEMONE_EMBED_PRO, "input": "test text"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["queue_name"] == QueueNames.EMBEDDING
    assert data["task_name"] == TaskNames.EMBEDDING
    assert data["task_id"] == "test-message-id"
    assert data["task_timestamp"] == 1234567890


def test_embeddings_async_result(client: TestClient, monkeypatch):
    """Test successful async embeddings result retrieval."""
    mock_result = {
        "model": Models.LEMONE_EMBED_PRO,
        "object": ObjectTypes.LIST,
        "data": [{
            "input": "test text",
            "index": 0,
            "object": ObjectTypes.EMBEDDING,
            "embedding": [0.1, 0.2, 0.3]
        }]
    }

    from app.services.broker import broker
    monkeypatch.setattr(broker, "get_result", lambda **kwargs: mock_result)

    response = client.get("/api/v1/embeddings/async/test-task-id")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == Models.LEMONE_EMBED_PRO
    assert data["object"] == ObjectTypes.LIST
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_similarity_async_creation(client: TestClient, monkeypatch):
    """Test successful async similarity task creation."""
    # Mock message with required attributes
    mock_message = MagicMock()
    mock_message.message_id = "test-message-id"
    mock_message.message_timestamp = 1234567890

    from app.workers.tasks import similarity_task
    monkeypatch.setattr(similarity_task, "send", lambda x: mock_message)

    response = client.post(
        "/api/v1/similarity/async",
        json={
            "model": Models.LEMONE_EMBED_PRO,
            "input": ["text1", "text2"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["queue_name"] == QueueNames.SIMILARITY
    assert data["task_name"] == TaskNames.SIMILARITY
    assert data["task_id"] == "test-message-id"
    assert data["task_timestamp"] == 1234567890


def test_similarity_async_result(client: TestClient, monkeypatch):
    """Test successful async similarity result retrieval."""
    mock_result = {
        "model": Models.LEMONE_EMBED_PRO,
        "object": ObjectTypes.LIST,
        "data": {
            "object": ObjectTypes.SIMILARITY,
            "data": [[1.0, 0.8], [0.8, 1.0]]
        }
    }

    from app.services.broker import broker
    monkeypatch.setattr(broker, "get_result", lambda **kwargs: mock_result)

    response = client.get("/api/v1/similarity/async/test-task-id")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == Models.LEMONE_EMBED_PRO
    assert data["object"] == ObjectTypes.LIST
    assert data["data"]["data"] == [[1.0, 0.8], [0.8, 1.0]]


def test_classification_async_creation(client: TestClient, monkeypatch):
    """Test successful async classification task creation."""
    # Mock message with required attributes
    mock_message = MagicMock()
    mock_message.message_id = "test-message-id"
    mock_message.message_timestamp = 1234567890

    from app.workers.tasks import classification_task
    monkeypatch.setattr(classification_task, "send", lambda x: mock_message)

    response = client.post(
        "/api/v1/classification/async",
        json={
            "model": ClassificationModels.LEMONE_ROUTER_L,
            "input": "test text"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["queue_name"] == QueueNames.CLASSIFICATION
    assert data["task_name"] == TaskNames.CLASSIFICATION
    assert data["task_id"] == "test-message-id"
    assert data["task_timestamp"] == 1234567890


def test_classification_async_result(client: TestClient, monkeypatch):
    """Test successful async classification result retrieval."""
    mock_result = {
        "model": ClassificationModels.LEMONE_ROUTER_L,
        "object": ObjectTypes.LIST,
        "data": [{
            "object": ObjectTypes.CLASSIFICATION,
            "label": "test",
            "score": 0.9,
            "index": 0
        }]
    }

    from app.services.broker import broker
    monkeypatch.setattr(broker, "get_result", lambda **kwargs: mock_result)

    response = client.get("/api/v1/classification/async/test-task-id")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == ClassificationModels.LEMONE_ROUTER_L
    assert data["object"] == ObjectTypes.LIST
    assert len(data["data"]) == 1
    assert data["data"][0]["label"] == "test"
    assert data["data"][0]["score"] == 0.9


def test_task_creation_error(client: TestClient, monkeypatch):
    """Test error handling in task creation."""
    from app.workers.tasks import embeddings_task

    def mock_send(data):
        raise Exception("Task creation failed")

    monkeypatch.setattr(embeddings_task, "send", mock_send)

    response = client.post(
        "/api/v1/embeddings/async",
        json={"model": Models.LEMONE_EMBED_PRO, "input": "test text"}
    )
    assert response.status_code == 500
    assert response.json()["error"] == "TaskInitalizationError"


def test_result_retrieval_error(client: TestClient, monkeypatch):
    """Test error handling in result retrieval."""
    from app.services.broker import broker

    def mock_get_result(**kwargs):
        raise Exception("Result not found")

    monkeypatch.setattr(broker, "get_result", mock_get_result)

    response = client.get("/api/v1/embeddings/async/non-existent-id")
    assert response.status_code == 404
    assert response.json()["error"] == "TaskNotFoundError"


def test_invalid_task_request_format(client: TestClient):
    """Test error handling for invalid task request format."""
    response = client.post(
        "/api/v1/embeddings/async",
        json={"invalid": "format"}
    )
    assert response.status_code == 422  # Validation error


def test_empty_task_input_validation(client: TestClient):
    """Test validation error on empty task input."""
    response = client.post(
        "/api/v1/embeddings/async",
        json={
            "model": Models.LEMONE_EMBED_PRO,
            "input": None  # None will trigger a validation error
        }
    )
    assert response.status_code == 422
