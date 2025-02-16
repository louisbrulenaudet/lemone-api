from .async_endpoints import router as async_router
from .sync_endpoints import router as sync_router

__all__ = ["sync_router", "async_router"]
