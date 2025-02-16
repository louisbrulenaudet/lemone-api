from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse

from app._exceptions import CoreError
from app.api.v1.router import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.name,
    description="API for processing french tax related texts using specialized models.",
    version="1.0.0",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(CoreError)
async def error_handler(request: Request, exc: CoreError) -> JSONResponse:
    """
    Custom exception handler for BusinessLogicError.
    Converts the error into a structured JSON response.
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "code": exc.code,
            "details": exc.details,
        },
    )
