"""Error format mandated by the platform.

Business errors are ``{"detail": {"reason": "..."}}``. Pydantic validation
failures are surfaced as HTTP 422 with the same envelope so the platform sees
one consistent shape.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """An error that maps to a documented HTTP status and reason."""

    def __init__(
        self,
        status_code: int,
        reason: str,
        *,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason
        self.headers = headers or {}
        self.extra = extra or {}

    def body(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"reason": self.reason}
        detail.update(self.extra)
        return {"detail": detail}


class BadRequest(AppError):
    def __init__(self, reason: str, **kwargs: Any) -> None:
        super().__init__(400, reason, **kwargs)


class Unauthorized(AppError):
    def __init__(self, reason: str = "invalid or missing credentials", **kwargs: Any) -> None:
        super().__init__(401, reason, **kwargs)


class Forbidden(AppError):
    def __init__(self, reason: str = "forbidden", **kwargs: Any) -> None:
        super().__init__(403, reason, **kwargs)


class NotFound(AppError):
    def __init__(self, reason: str = "not found", **kwargs: Any) -> None:
        super().__init__(404, reason, **kwargs)


class PayloadTooLarge(AppError):
    def __init__(self, reason: str, **kwargs: Any) -> None:
        super().__init__(413, reason, **kwargs)


class UnprocessableEntity(AppError):
    def __init__(self, reason: str, **kwargs: Any) -> None:
        super().__init__(422, reason, **kwargs)


class ServiceUnavailable(AppError):
    def __init__(self, reason: str = "service temporarily unavailable", **kwargs: Any) -> None:
        super().__init__(503, reason, **kwargs)
