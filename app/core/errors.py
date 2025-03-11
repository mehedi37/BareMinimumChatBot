from typing import Any, Dict, Optional
from fastapi import status


class AppBaseException(Exception):
    """
    Base exception for application errors
    """
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An unexpected error occurred",
        code: str = "server_error",
        errors: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.errors = errors
        super().__init__(self.detail)


class NotFoundException(AppBaseException):
    """
    Exception raised when a resource is not found
    """
    def __init__(
        self,
        detail: str = "Resource not found",
        code: str = "not_found",
        errors: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            code=code,
            errors=errors,
        )


class BadRequestException(AppBaseException):
    """
    Exception raised when a request is invalid
    """
    def __init__(
        self,
        detail: str = "Invalid request",
        code: str = "bad_request",
        errors: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            code=code,
            errors=errors,
        )


class UnauthorizedException(AppBaseException):
    """
    Exception raised when authentication fails
    """
    def __init__(
        self,
        detail: str = "Authentication failed",
        code: str = "unauthorized",
        errors: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            code=code,
            errors=errors,
        )


class ForbiddenException(AppBaseException):
    """
    Exception raised when a user doesn't have permission
    """
    def __init__(
        self,
        detail: str = "Permission denied",
        code: str = "forbidden",
        errors: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            code=code,
            errors=errors,
        )


class ChatbotException(AppBaseException):
    """
    Exception raised when there's an error with the chatbot
    """
    def __init__(
        self,
        detail: str = "Chatbot processing error",
        code: str = "chatbot_error",
        errors: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            code=code,
            errors=errors,
        )