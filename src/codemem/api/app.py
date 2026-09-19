"""FastAPI application: the Add / Search contract.

Contract points that are easy to get wrong and are enforced here:

* Add returns HTTP 200 with ``success: true`` only after memory is durably
  stored and immediately searchable;
* the three identifiers are echoed verbatim;
* ``data`` is always present, items are non-empty strings, the response never
  exceeds ``top_k``;
* business errors use ``{"detail": {"reason": ...}}``;
* ``/health`` is unauthenticated and returns 2xx.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from ..core.config import Settings
from ..core.errors import AppError, ServiceUnavailable
from ..core.logging import get_logger, log_ctx, setup_logging
from ..core.schemas import (
    AddRequest,
    AddResponse,
    HealthResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
)
from ..index.store import Store
from ..add.pipeline import AddPipeline
from ..search.service import SearchPipeline
from .security import require_auth

log = get_logger("codemem.api")


class Container:
    """Process-wide singletons, built once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = Store(settings)
        self.add = AddPipeline(settings, self.store)
        self.search = SearchPipeline(settings, self.store)
        self.ready = True

    def close(self) -> None:
        self.store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = Container(settings)
        log.info(
            "service started",
            extra={
                "ctx": {
                    "version": __version__,
                    "data_dir": str(settings.data_dir),
                    "auth": bool(settings.api_key),
                    "llm": settings.llm_enabled,
                }
            },
        )
        try:
            yield
        finally:
            app.state.container.close()

    app = FastAPI(
        title="codemem",
        version=__version__,
        description="Code-memory Add/Search service (Agent Memory Challenge, Coding track)",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------- errors --

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body(), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # One consistent error envelope; the platform documents 422 for invalid
        # payloads with structured field details.
        errors = []
        for err in exc.errors():
            errors.append(
                {
                    "loc": [str(p) for p in err.get("loc", ())],
                    "msg": err.get("msg", "invalid value"),
                    "type": err.get("type", "value_error"),
                }
            )
        return JSONResponse(
            status_code=422,
            content={"detail": {"reason": "request validation failed", "errors": errors}},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Never leak internals; the platform only needs a stable envelope.
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"detail": {"reason": "internal error"}},
        )

    # -------------------------------------------------------------- routes --

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        container: Container = app.state.container
        users, memories = container.store.counts()
        return HealthResponse(
            status="ok", version=__version__, users=users, memories=memories
        )

    @app.post("/add", response_model=AddResponse)
    async def add(request: Request, payload: AddRequest) -> AddResponse:
        container: Container = app.state.container
        if not container.ready:
            raise ServiceUnavailable("service is not ready")
        require_auth(request, container.settings.api_key)

        if len(payload.messages) > container.settings.max_messages_per_add:
            raise AppError(
                413,
                f"too many messages (max {container.settings.max_messages_per_add})",
            )

        t0 = time.monotonic()
        try:
            outcome, degraded = container.add.handle(
                request_id=payload.request_id,
                user_id=payload.user_id,
                session_id=payload.session_id,
                messages=payload.messages,
            )
        except AppError:
            raise
        except Exception:
            log.exception("add failed")
            raise ServiceUnavailable("add failed; retry with the same request_id")

        log_ctx(
            log,
            logging.INFO,
            "add served",
            request_id=payload.request_id,
            user_id=payload.user_id,
            duplicate=outcome.duplicate,
            memories=outcome.memories_written,
            degraded=degraded,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return AddResponse(
            success=True,
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(request: Request, payload: SearchRequest) -> SearchResponse:
        container: Container = app.state.container
        if not container.ready:
            raise ServiceUnavailable("service is not ready")
        require_auth(request, container.settings.api_key)

        if payload.top_k < 1:
            raise AppError(400, "top_k must be >= 1")

        items = container.search.handle(
            user_id=payload.user_id,
            query=payload.query,
            options=payload.options,
            top_k=payload.top_k,
        )
        # Defensive clamp: exceeding top_k is a contract error, not something
        # the platform truncates for us.
        items = items[: payload.top_k]
        return SearchResponse(
            data=[
                SearchItem(
                    id=f"mem_{item.memory_id}",
                    content=item.content,
                    score=item.score,
                    created_at=item.created_at,
                )
                for item in items
                if item.content
            ]
        )

    # ------------------------------------------------------------ admin --

    @app.delete("/admin/users/{user_id:path}")
    async def delete_user(request: Request, user_id: str) -> dict:
        """Retention control: erase every trace of a user_id on request."""
        container: Container = app.state.container
        require_auth(request, container.settings.api_key)
        removed = container.store.delete_user(user_id)
        return {"success": True, "user_id": user_id, "removed": removed}

    return app


app = create_app()
