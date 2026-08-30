"""Structured error payload and domain exception types.

This module owns the single error contract shared across the platform. Every
error the API returns - request-validation failures, domain errors raised by
feature services, a database-unreachable condition, or an unexpected internal
error - is serialized into one shape by the exception handlers registered in
``app.main``:

    {"error": {"code": str, "message": str, "status": int, "details": object}}

Centralizing the contract here means feature services never build ad-hoc error
bodies; they raise :class:`AppError` (or a subclass) and the central handlers
render it.

Requirements:
    15.3 - error responses return a structured payload containing a
           human-readable message and a status code.
    16.4 - the database-unreachable path returns a service-unavailable status
           with a retry message rather than an unhandled failure (the payload
           used by that handler is built here).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """The inner ``error`` object of the structured payload.

    Attributes:
        code: A stable, machine-readable identifier for the error class
            (e.g. ``"VALIDATION_ERROR"``).
        message: A human-readable description safe to show to a client
            (Req 15.3).
        status: The HTTP status code associated with the error (Req 15.3).
        details: Optional structured context (defaults to an empty object so
            the field is always present).
    """

    code: str
    message: str
    status: int
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    """Top-level error response body.

    Serializes to ``{"error": {"code", "message", "status", "details"}}`` - the
    single error contract described in the design's Error Handling section
    (Req 15.3).
    """

    error: ErrorDetail

    @classmethod
    def build(
        cls,
        *,
        code: str,
        message: str,
        status: int,
        details: dict[str, Any] | None = None,
    ) -> "ErrorPayload":
        """Construct a payload from its parts.

        ``details`` defaults to an empty object so the serialized shape always
        contains all four keys.
        """

        return cls(
            error=ErrorDetail(
                code=code,
                message=message,
                status=status,
                details=details or {},
            )
        )


class AppError(Exception):
    """Base class for domain errors raised by feature services.

    Raising an ``AppError`` (or a subclass) anywhere in the request lifecycle is
    translated by the central exception handler into the structured
    :class:`ErrorPayload` and returned with this error's HTTP ``status``. Feature
    services should raise this instead of returning bespoke error bodies.

    Args:
        code: Stable machine-readable error identifier.
        message: Human-readable, client-safe message (Req 15.3).
        status: HTTP status code to return (defaults to ``400``).
        details: Optional structured context included in the payload.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details: dict[str, Any] = details or {}

    def to_payload(self) -> ErrorPayload:
        """Render this error as the structured :class:`ErrorPayload`."""

        return ErrorPayload.build(
            code=self.code,
            message=self.message,
            status=self.status,
            details=self.details,
        )
