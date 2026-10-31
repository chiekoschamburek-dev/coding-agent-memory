"""HTTP layer.

Contract-faithful to api_guide.md: the Add response echoes exactly the three
identifiers, Search always returns a ``data`` array (empty rather than absent),
and no field is added to those payloads. Operational detail goes to separate
diagnostic endpoints and to logs, so the graded responses stay clean.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .common.config import Settings, get_settings
from .common.embeddings import get_model_limiter
from .common.guard import get_guard
from .common.llm import LLMError, LLMUnavailable, get_llm
from .service import IngestionInProgress, MemoryService, ValidationError
from .common.store import Store
from .common.tokens import get_counter

logger = logging.getLogger("codemem")


# --------------------------------------------------------------------------
# request / response models
# --------------------------------------------------------------------------

# Field *presence* is enforced here (a missing required key is a 422, the
# framework-standard signal for a malformed body). Field *values* are typed
# permissively on purpose: they are validated in ``MemoryService`` so that every
# rejected value produces the same 400 shape with a specific, published reason,
# which plan.md A1 requires ("resource limits and error responses are published,
# never silently truncated"). Splitting the two keeps one code path for all
# value errors instead of duplicating rules across the model and the service.


class MessageIn(BaseModel):
    role: Any = None
    content: Any = None
    timestamp: Any = None

    model_config = {"extra": "allow"}


class AddRequest(BaseModel):
    request_id: Any = None
    messages: Any = None
    user_id: Any = None
    session_id: Any = None


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: Any = None
    options: Any = None
    user_id: Any = None
    top_k: Any = None


class SearchItem(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[SearchItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# app wiring
# --------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(cfg)
        # Recover any ingestion interrupted by a previous process (plan.md A1).
        try:
            recovered = store.recover_stale()
            if recovered:
                logger.warning("recovered %d stale ingestions", recovered)
        except Exception:  # pragma: no cover - startup resilience
            logger.exception("stale recovery failed")
        app.state.store = store
        app.state.service = MemoryService(store, cfg)
        app.state.settings = cfg
        app.state.counter = get_counter(cfg)
        logger.info(
            "codemem %s ready (db=%s, tokenizer=%s, embeddings=%s)",
            __version__,
            cfg.db_path,
            app.state.counter.mode,
            "on" if cfg.embeddings_enabled else "off",
        )
        yield

    app = FastAPI(
        title="Code Memory System",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    # -- authentication --------------------------------------------------

    def require_auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        """Bearer / Token / X-API-Key check.

        Disabled when no key is configured, so a local run needs no setup; when
        a key *is* configured it is enforced on every graded endpoint. Health
        stays open (plan.md 7.3) so an orchestrator can probe liveness without
        credentials.
        """
        expected = cfg.api_key
        if not expected:
            return
        presented: str | None = None
        if authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() in {"bearer", "token"}:
                presented = value.strip()
        if presented is None and x_api_key:
            presented = x_api_key.strip()
        if presented != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

    # -- error shape -----------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _request_shape_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Malformed body (missing key, wrong JSON type) -> 422, same shape as 400."""
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        detail = first.get("msg", "invalid request body")
        return JSONResponse(
            status_code=422,
            content={"detail": f"{location}: {detail}" if location else detail},
        )

    @app.exception_handler(ValidationError)
    async def _validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.detail})

    @app.exception_handler(IngestionInProgress)
    async def _in_progress_handler(request: Request, exc: IngestionInProgress) -> JSONResponse:
        # Retryable: the platform applies bounded backoff (rules.md).
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "retryable": True},
            headers={"Retry-After": "2"},
        )

    @app.exception_handler(LLMUnavailable)
    async def _llm_unavailable_handler(request: Request, exc: LLMUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "model backend unavailable", "retryable": True},
        )

    # -- endpoints -------------------------------------------------------

    @app.post("/add", response_model=AddResponse, dependencies=[Depends(require_auth)])
    def add(payload: AddRequest, request: Request) -> AddResponse:
        """Synchronous route on purpose (FIXES.md P1.2).

        ``service.add`` is blocking: SQLite writes, CPU chunking, and possibly a
        synchronous embedding call. Declaring this ``async`` would run all of
        that on the event loop, so one slow Add would stall every other request
        and the health check. As a sync route FastAPI runs it in its worker
        threadpool, which keeps the loop free; model concurrency is bounded
        separately by ``ModelCallLimiter``.
        """
        service: MemoryService = request.app.state.service
        started = time.monotonic()
        outcome = service.add(
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
            raw_messages=payload.messages,
        )
        # Log counts and timing only -- never request content (plan.md 7.2).
        logger.info(
            json.dumps(
                {
                    "event": "add",
                    "user_id": payload.user_id,
                    "request_id": payload.request_id,
                    "messages": len(payload.messages) if isinstance(payload.messages, list) else 0,
                    "mode": outcome.mode,
                    "dedup": outcome.deduplicated,
                    "elapsed_ms": outcome.elapsed_ms,
                    "wall_ms": int((time.monotonic() - started) * 1000),
                }
            )
        )
        return AddResponse(
            success=True,
            request_id=outcome.request_id,
            user_id=outcome.user_id,
            session_id=outcome.session_id,
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(require_auth)],
    )
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        # Same reasoning as /add: the search pipeline is blocking (SQLite reads,
        # optional embedding call), so it runs in FastAPI's worker threadpool
        # rather than on the event loop.
        service: MemoryService = request.app.state.service
        outcome = service.search(
            user_id=payload.user_id,
            query=payload.query,
            top_k=payload.top_k,
            options=payload.options,
        )
        logger.info(
            json.dumps(
                {
                    "event": "search",
                    "user_id": payload.user_id,
                    "options": len(payload.options) if isinstance(payload.options, list) else 0,
                    "selected": len(outcome.data),
                    "tokens": outcome.diagnostics.get("tokens_used"),
                    "applicability": outcome.diagnostics.get("applicability"),
                    "elapsed_ms": outcome.diagnostics.get("elapsed_ms"),
                    "degraded": outcome.diagnostics.get("degraded"),
                }
            )
        )
        return SearchResponse(data=[SearchItem(**item) for item in outcome.data])

    # -- diagnostics (not part of the graded contract) -------------------

    @app.get("/health")
    def health(request: Request) -> dict:
        # Sync route: ``store.health()`` is a blocking DB read, and this
        # endpoint must stay responsive while a slow Add occupies workers
        # (FIXES.md P1.2).
        store: Store = request.app.state.store
        guard = get_guard(cfg)
        db = store.health()
        healthy = db.get("db") == "ok"
        return {
            "status": "ok" if healthy else "degraded",
            "version": __version__,
            "db": db,
            "guard": guard.snapshot(),
            "embeddings": cfg.embeddings_enabled,
            "tokenizer": get_counter(cfg).mode,
            "model_calls": get_model_limiter(cfg).stats(),
            "uptime_s": int(time.monotonic() - _STARTED_AT),
        }

    @app.get("/ready")
    def ready(request: Request) -> JSONResponse:
        """Readiness distinct from liveness: the DB must answer."""
        db = request.app.state.store.health()
        ok = db.get("db") == "ok"
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok})

    @app.get("/diagnostics", dependencies=[Depends(require_auth)])
    async def diagnostics(request: Request) -> dict:
        """Operational counters. Contains no request content."""
        settings: Settings = request.app.state.settings
        return {
            "version": __version__,
            "guard": get_guard(settings).snapshot(),
            "config": {
                "add_sync_budget_s": settings.add_sync_budget_s,
                "search_budget_p95_s": settings.search_budget_p95_s,
                "search_hard_budget_s": settings.search_hard_budget_s,
                "token_budget_total": settings.token_budget_total,
                "token_budget_target": settings.token_budget_target,
                "top_k_max": settings.top_k_max,
                "search_max_items": settings.search_max_items,
                "rank_boost_scale": settings.rank_boost_scale,
                "applicability_enabled": settings.applicability_enabled,
                "applicability_weight": settings.applicability_weight,
                "query_denoise": settings.query_denoise,
                "pool_per_channel": settings.pool_per_channel,
                "pool_total": settings.pool_total,
                "model": settings.llm_model,
                "embeddings_enabled": settings.embeddings_enabled,
                "rerank_enabled": bool(settings.rerank_base_url) and not settings.transferability_enabled,
                "transferability_enabled": settings.transferability_enabled,
                "transferability_candidates": settings.transferability_max_candidates,
                "transferability_budget_s": settings.transferability_budget_s,
                "rerank_candidates": settings.rerank_candidates,
                "auth_required": bool(settings.api_key),
            },
        }

    @app.post("/admin/purge", dependencies=[Depends(require_auth)])
    async def purge(payload: dict, request: Request) -> dict:
        """Delete one isolation scope (retention obligation, plan.md 7.3)."""
        user_id = payload.get("user_id")
        if not user_id or not isinstance(user_id, str):
            raise HTTPException(status_code=400, detail="user_id is required")
        removed = request.app.state.store.purge_user(user_id)
        logger.info(json.dumps({"event": "purge", "user_id": user_id, "removed": removed}))
        return {"purged": user_id, "removed": removed}

    @app.post("/admin/llm-probe", dependencies=[Depends(require_auth)])
    async def llm_probe(payload: dict | None = None) -> dict:
        """Verify the model backend before spending tokens on real work.

        Default is a local-only check that sends nothing. ``{"live": true}``
        sends one minimal request.
        """
        live = bool((payload or {}).get("live", False))
        client = get_llm(cfg)
        try:
            return client.probe(live=live)
        except LLMError as exc:
            return {"status": "error", "detail": str(exc)}

    @app.post("/admin/embed-probe", dependencies=[Depends(require_auth)])
    async def embed_probe(payload: dict | None = None) -> dict:
        """Verify the embedding backend. Same contract as ``llm-probe``.

        Local by default (sends nothing); ``{"live": true}`` sends one minimal
        request and reports the returned dimension against ``CODEMEM_EMBED_DIM``.
        """
        from .common.embeddings import get_embedder

        live = bool((payload or {}).get("live", False))
        embedder = get_embedder(cfg, cfg.embed_cache_path)
        return embedder.probe(live=live)

    return app


_STARTED_AT = time.monotonic()
