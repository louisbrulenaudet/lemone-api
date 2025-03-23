import sys
from unittest.mock import MagicMock

# Create a complete mock dramatiq structure before any imports
mock_dramatiq = MagicMock()
mock_dramatiq.middleware = MagicMock()
mock_dramatiq.middleware.Middleware = type('MockMiddleware', (), {
    '__init__': lambda *args, **kwargs: None,
    'after_process_boot': lambda *args, **kwargs: None,
    'before_consumer_thread_shutdown': lambda *args, **kwargs: None,
    'after_consumer_thread_shutdown': lambda *args, **kwargs: None,
})

# Create mock broker
mock_broker = MagicMock()

# Create mock cache decorator that just returns the function unchanged
def mock_cached(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

# Patch the modules before they're imported
sys.modules["dramatiq"] = mock_dramatiq
sys.modules["dramatiq.middleware"] = mock_dramatiq.middleware
sys.modules["dramatiq_redis"] = MagicMock()
sys.modules["app.services.broker"] = MagicMock(broker=mock_broker)
sys.modules["aiocache"] = MagicMock()
sys.modules["aiocache"].cached = mock_cached

# Now we can safely import pytest and other modules
import pytest
from fastapi.testclient import TestClient

from app._enums import ObjectTypes
from app.models.models import Classification, Embedding, Similarity

# Create other mock objects
mock_registry = MagicMock()
mock_redis = MagicMock()
mock_model = MagicMock()

# Configure mock model responses
mock_model.encode.return_value = [
    Embedding(
        input="test text",
        index=0,
        object=ObjectTypes.EMBEDDING,
        embedding=[0.1, 0.2, 0.3]
    )
]

mock_model.similarity.return_value = Similarity(
    object=ObjectTypes.SIMILARITY,
    data=[[1.0, 0.8], [0.8, 1.0]]
)

mock_model.classify.return_value = [
    Classification(
        object=ObjectTypes.CLASSIFICATION,
        label="test",
        score=0.9,
        index=0
    )
]

# Configure registry mock
mock_registry.get_model.return_value = mock_model

# Additional patches
patches = [
    ("app.core.config.settings.registry", mock_registry),
    ("redis.Redis", mock_redis),
    ("aiocache.backends.redis.RedisCache", MagicMock()),
]

# Apply patches
_mocks = []
for target, mock_obj in patches:
    patcher = pytest.MonkeyPatch()
    patcher.setattr(target, mock_obj, raising=False)
    _mocks.append(patcher)

def pytest_sessionfinish(session, exitstatus):
    """Cleanup patches at end of test session."""
    # Clear the monkeypatched attributes
    for mock in _mocks:
        mock.undo()

    # Remove mocked modules
    for mod in ["dramatiq", "dramatiq.middleware", "dramatiq_redis", "app.services.broker", "aiocache"]:
        sys.modules.pop(mod, None)

@pytest.fixture
def client():
    """Create a test client."""
    from app.main import app
    return TestClient(app)

@pytest.fixture
def mock_model_fixture():
    """Return the mock model."""
    return mock_model
