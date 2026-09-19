"""Authentication.

The platform supports Token, Bearer, or X-Api-Key. When ``CODEMEM_API_KEY`` is
unset the service runs unauthenticated, which the rules permit only for public
smoke; the health endpoint is always unauthenticated.
"""

from __future__ import annotations

import hmac

from fastapi import Request

from ..core.errors import Unauthorized


def _presented_token(request: Request) -> tuple[str | None, str]:
    header = request.headers.get("authorization")
    if header:
        parts = header.split(None, 1)
        if len(parts) == 2:
            scheme, value = parts[0].lower(), parts[1].strip()
            if scheme in {"bearer", "token"}:
                return value, scheme
            return value, scheme
        if len(parts) == 1:
            return parts[0].strip(), "bare"
    for name in ("x-api-key", "api-key", "x-api-token"):
        value = request.headers.get(name)
        if value:
            return value.strip(), name
    return None, "none"


def require_auth(request: Request, expected: str | None) -> None:
    if not expected:
        return
    presented, scheme = _presented_token(request)
    if not presented:
        raise Unauthorized("missing credentials", headers={"WWW-Authenticate": "Bearer"})
    if not hmac.compare_digest(presented, expected):
        raise Unauthorized("invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    del scheme
