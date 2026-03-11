"""Request middleware: correlation ID, rate-limit headers, error handlers."""

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from engram.core.errors import EngramError
from engram.core.tracing import set_correlation_id
from engram.models import generate_correlation_id

logger = structlog.get_logger()


# --- Correlation ID + Rate-Limit Headers ---


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Attach correlation_id to every request/response. Add rate-limit headers."""

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        correlation_id = generate_correlation_id()
        request.state.correlation_id = correlation_id
        set_correlation_id(correlation_id)

        response = await call_next(request)

        response.headers["X-Correlation-ID"] = correlation_id

        rate_info = getattr(request.state, "rate_info", None)
        if rate_info:
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

        return response


# --- Error Handlers ---


def register_error_handlers(app: FastAPI) -> None:
    """Register exception handlers that produce the standard error schema."""

    @app.exception_handler(EngramError)
    async def engram_error_handler(
        request: Request, exc: EngramError
    ) -> JSONResponse:
        correlation_id = getattr(
            request.state, "correlation_id", generate_correlation_id()
        )
        body: dict = {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "status": exc.status,
                "correlation_id": correlation_id,
            }
        }
        if exc.existing_id is not None:
            body["error"]["existing_id"] = exc.existing_id

        headers: dict[str, str] = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)

        return JSONResponse(
            status_code=exc.status, content=body, headers=headers
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        correlation_id = getattr(
            request.state, "correlation_id", generate_correlation_id()
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": _http_status_to_code(exc.status_code),
                    "message": exc.detail or "An error occurred",
                    "status": exc.status_code,
                    "correlation_id": correlation_id,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        correlation_id = getattr(
            request.state, "correlation_id", generate_correlation_id()
        )
        errors = exc.errors()
        parts = []
        for e in errors:
            loc = e.get("loc", ())
            field = loc[-1] if loc else "unknown"
            parts.append(f"{field}: {e['msg']}")
        msg = "; ".join(parts) if parts else "Invalid request"

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_INPUT",
                    "message": msg,
                    "status": 400,
                    "correlation_id": correlation_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        correlation_id = getattr(
            request.state, "correlation_id", generate_correlation_id()
        )
        logger.error(
            "unhandled_error",
            correlation_id=correlation_id,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "status": 500,
                    "correlation_id": correlation_id,
                }
            },
        )


_HTTP_CODE_MAP: dict[int, str] = {
    400: "INVALID_INPUT",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


def _http_status_to_code(status: int) -> str:
    """Map HTTP status to error code string."""
    return _HTTP_CODE_MAP.get(status, f"HTTP_{status}")
