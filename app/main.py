from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse

from app._exceptions import CoreError
from app.api.v1.router import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.name,
    description="API for processing french tax related texts using specialized models.",
    version="0.1.1",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(CoreError)
async def error_handler(request: Request, exc: CoreError) -> JSONResponse:
    """
    Custom exception handler for BusinessLogicError.
    Converts the error into a structured JSON response.
    """
    # Map specific exceptions to status codes
    status_codes = {
        "TaskNotFoundError": 404,
        "TaskInitalizationError": 500,
        "ModelRegistryNotFoundError": 404,
    }

    # Default to 400 if not specified
    status_code = status_codes.get(exc.__class__.__name__, 400)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "code": exc.code,
            "details": exc.details,
        },
    )
