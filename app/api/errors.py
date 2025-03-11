from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Any, Dict, List, Union
from loguru import logger

from app.core.errors import AppBaseException


async def app_exception_handler(request: Request, exc: AppBaseException) -> JSONResponse:
    """
    Handle custom application exceptions
    """
    logger.error(f"App exception: {exc.detail}, code: {exc.code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code": exc.code,
            "message": exc.detail,
            "errors": exc.errors or {},
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle validation errors
    """
    errors: List[Dict[str, Any]] = []
    for error in exc.errors():
        location = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "location": location,
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(f"Validation error: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "code": "validation_error",
            "message": "Request validation failed",
            "errors": errors,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions
    """
    logger.exception("Unexpected error occurred")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": "server_error",
            "message": "An unexpected error occurred",
        },
    )


def add_exception_handlers(app: FastAPI) -> None:
    """
    Add exception handlers to the FastAPI app
    """
    app.add_exception_handler(AppBaseException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)