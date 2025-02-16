from fastapi import APIRouter

from app.api.v1.endpoints import async_router, sync_router

router = APIRouter()

# Include all routers
router.include_router(sync_router)
router.include_router(async_router)
