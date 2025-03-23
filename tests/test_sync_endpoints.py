import numpy as np
import pytest
from fastapi.testclient import TestClient

from app._enums import ClassificationModels, Models, ObjectTypes


def test_embeddings_endpoint_success(client: TestClient, mock_model_fixture):
    """Test successful embeddings generation."""
    response = client.post(
        "/api/v1/embeddings",
        json={"model": Models.LEMONE_EMBED_PRO, "input": "test text"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == Models.LEMONE_EMBED_PRO
    assert data["object"] == ObjectTypes.LIST
    assert len(data["data"]) == 1
    assert data["data"][0]["input"] == "test text"
    assert data["data"][0]["index"] == 0
    assert data["data"][0]["object"] == ObjectTypes.EMBEDDING
    assert pytest.approx(data["data"][0]["embedding"]) == [0.1, 0.2, 0.3]


def test_embeddings_endpoint_compute_error(client: TestClient, mock_model_fixture):
    """Test embeddings endpoint with compute error."""
    mock_model_fixture.encode.side_effect = ValueError("Compute error")
    response = client.post(
        "/api/v1/embeddings",
        json={"model": Models.LEMONE_EMBED_PRO, "input": "test text"},
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["error"] == "EmbeddingComputeError"
    assert error_response["details"] == "Compute error"


def test_similarity_endpoint_success(client: TestClient, mock_model_fixture):
    """Test successful similarity computation."""
    response = client.post(
        "/api/v1/similarity",
        json={"model": Models.LEMONE_EMBED_PRO, "input": ["text1", "text2"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == Models.LEMONE_EMBED_PRO
    assert data["object"] == ObjectTypes.LIST
    # Convert similarity matrix to numpy array for comparison
    similarity_matrix = np.array(data["data"]["data"])
    expected_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    assert np.allclose(similarity_matrix, expected_matrix)
    assert data["data"]["object"] == ObjectTypes.SIMILARITY


def test_similarity_endpoint_compute_error(client: TestClient, mock_model_fixture):
    """Test similarity endpoint with compute error."""
    mock_model_fixture.similarity.side_effect = ValueError("Compute error")
    response = client.post(
        "/api/v1/similarity",
        json={"model": Models.LEMONE_EMBED_PRO, "input": ["text1", "text2"]},
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["error"] == "EmbeddingComputeError"
    assert error_response["details"] == "Compute error"


def test_classification_endpoint_success(client: TestClient, mock_model_fixture):
    """Test successful classification."""
    response = client.post(
        "/api/v1/classification",
        json={"model": ClassificationModels.LEMONE_ROUTER_L, "input": "test text"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == ClassificationModels.LEMONE_ROUTER_L
    assert data["object"] == ObjectTypes.LIST
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == ObjectTypes.CLASSIFICATION
    assert data["data"][0]["label"] == "test"
    assert pytest.approx(data["data"][0]["score"]) == 0.9
    assert data["data"][0]["index"] == 0


def test_classification_endpoint_compute_error(client: TestClient, mock_model_fixture):
    """Test classification endpoint with compute error."""
    mock_model_fixture.classify.side_effect = ValueError("Compute error")
    response = client.post(
        "/api/v1/classification",
        json={"model": ClassificationModels.LEMONE_ROUTER_L, "input": "test text"},
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["error"] == "ClassificationComputeError"
    assert error_response["details"] == "Compute error"


def test_invalid_request_format(client: TestClient):
    """Test error handling for invalid request format."""
    response = client.post("/api/v1/embeddings", json={"invalid": "format"})
    assert response.status_code == 422  # Validation error


def test_empty_input_validation(client: TestClient):
    """Test validation error on empty input."""
    response = client.post(
        "/api/v1/embeddings",
        json={
            "model": Models.LEMONE_EMBED_PRO,
            "input": None,  # None will trigger a validation error
        },
    )
    assert response.status_code == 422
