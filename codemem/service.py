"""Add / Search orchestration.

Add flow:

    validate -> idempotency -> reserve key -> chunk (CPU) -> COMMIT (chunks)
    -> finalise this session (coding sync mode) -> return success

The ordering matters: the idempotency key is reserved *before* any expensive
work so concurrent duplicates cannot both proceed, and the success response is
sent only after the chunk transaction has committed, so ``success: true``
always means "persisted" as api_guide.md requires.

Coding deployment acknowledges each Add only after extracting its session.
Same-session mutations serialize; independent sessions share bounded model
slots. Card replacement publishes atomically and Search uses a WAL snapshot.
Extraction failures retain durable chunks for idempotent retries, return 503,
and never inject unfinished raw content into Search.

In legacy asynchronous mode, cards are extracted per **session**. A session is considered
closed the moment an Add with a *different* session_id arrives (``session_id``
change is the only end-of-session signal the API offers); Search finalises
whatever is still pending as a backstop, so the last session of a scope also
gets its cards. Chunks are the persisted minimal facts -- raw messages are
never stored, and only cards are embedded.
"""

from __future__ import annotations

import json
import re
import threading
import time
import weakref
from dataclasses import dataclass, field
from functools import wraps

from .search.applicability import ListwiseApplicabilityJudge
from .common.chunking import build_chunks
from .common.config import Settings, get_settings
from .common.embeddings import get_embedder, get_model_limiter, pack_vector
from .search.transferability import MinimalTransferabilityJudge
from .add.experience import (
    build_cards,
    build_extraction_prompt,
    card_admission_problems,
    card_anchors,
    card_entities,
    card_field_terms,
    card_retrieval_text,
    card_to_payload,
    enrich_card,
    extraction_system_prompt,
    is_setup_chunk,
    render_experience,
)
from .common.guard import MODE_FULL, ModuleBudget, get_guard
from .common.llm import LLMUnavailable, get_llm
from .add.minimal import (
    VERSION as MINIMAL_VERSION,
)
from .add.minimal import (
    build_evidence as build_minimal_evidence,
)
from .add.minimal import (
    build_prompt as build_minimal_prompt,
)
from .add.minimal import (
    is_minimal,
    render_minimal,
)
from .search.rerank import HttpReranker
from .search.retrieval import SearchEngine, StoreExperienceVectorIndex
from .common.store import (
    IngestionConflict,
    IngestionInProgress,
    MessageInput,
    Store,
    content_hash,
    experience_id_for,
    now_iso,
)
from .common.tokens import get_counter

_FINAL_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)


def _final_diff_files(chunks) -> frozenset:
    """Files of the session's final staged diff, scanned from chunk texts.

    300Lite-style sessions end with ``git diff --cached``; the staged diff is
    the mechanical ground truth of what survived.  Returns an empty set when
    no such chunk exists (the clamp then no-ops -- soft by design, the final
    diff is a harness convention, not a guarantee).
    """

    for chunk in reversed(chunks):
        text = getattr(chunk, "text", "") or ""
        if "diff --git a/" not in text:
            continue
        found = {m.group(1) for m in _FINAL_DIFF_FILE_RE.finditer(text)}
        return frozenset(f.rsplit("/", 1)[-1] for f in found)
    return frozenset()


def _clamp_files_to_final_diff(card, final_files: frozenset):
    """Drop card files that the session's final diff does not contain.

    Measured on the 300Lite corpus: 23/110 cards carried files beyond the
    final diff (explore-only touches); those pollute the index locate channel
    and the injected scope prior.  When the card shares no file with the
    final diff at all, keep the original (parse failure must not gut a card).
    """
    if card.is_minimal or not final_files or not card.files:
        return card
    clamped = tuple(f for f in card.files if f.rsplit("/", 1)[-1] in final_files)
    if not clamped or clamped == card.files:
        return card
    from dataclasses import replace
    return replace(card, files=clamped)


class ValidationError(Exception):
    """Request rejected by A1 checks. Published, never silent."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass
class AddOutcome:
    success: bool
    request_id: str
    user_id: str
    session_id: str
    mode: str
    deduplicated: bool = False
    steps: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0


@dataclass
class SearchOutcome:
    data: list[dict]
    diagnostics: dict


class ModelCallBudget:
    """Hard cap on LLM calls for one session finalisation."""

    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


_CONTENT_WORD = re.compile(r"[a-z0-9_]{3,}")


def _content_words(text: str) -> list[str]:
    """Distinct low-impedance query terms for the pending-chunk fallback."""
    return list(dict.fromkeys(_CONTENT_WORD.findall(text.lower())))


def _serialize_scope(method):
    """Serialize session mutations; legacy async mode retains its scope lock."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        user_id = kwargs.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return method(self, *args, **kwargs)
        key = user_id
        if self.settings.add_finalize_sync:
            # Independent sessions may extract concurrently. Search gets a WAL
            # snapshot and never waits for an unrelated session's model call.
            session_id = kwargs.get("session_id")
            if method.__name__ not in {"add", "_finalize_session"} or not isinstance(session_id, str):
                return method(self, *args, **kwargs)
            key = (user_id, session_id)
        with self._scope_lock(key):
            return method(self, *args, **kwargs)
    return wrapped


class MemoryService:
    def __init__(
        self,
        store: Store,
        settings: Settings | None = None,
        search_engine: SearchEngine | None = None,
        llm_client=None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self._scope_locks = weakref.WeakValueDictionary()
        self._scope_locks_guard = threading.Lock()

        # Background finalisation (contract: persisted => immediately
        # retrievable).  At most one worker per user scope; Search's backstop
        # remains the fallback for anything the worker did not finish.
        self._bg_finalize_users: set[str] = set()
        self._bg_finalize_lock = threading.Lock()

        # The LLM client is created lazily and injectable: experience extraction
        # is best-effort enrichment, so a missing/failing provider must degrade
        # rather than break Add. Tests inject a fake here.
        self._llm = llm_client

        # P2.8 fix: Create embedder once at init; shared by Search and Add enrichment
        self.embedder = None
        if self.settings.embeddings_enabled:
            self.embedder = get_embedder(self.settings, self.settings.embed_cache_path)

        if search_engine is not None:
            self.search_engine = search_engine
        else:
            # Wire the card vector channel only when embeddings are configured;
            # the SearchEngine falls back to a null index otherwise, which is the
            # lex-only baseline.  Chunks are never embedded.
            card_vector_index = None
            if self.embedder is not None:
                card_vector_index = StoreExperienceVectorIndex(
                    self.store, self.embedder, self.settings.embed_model,
                    timeout_s=self.settings.embed_timeout_s,
                )
            llm = self._llm or get_llm(self.settings)
            judge = ListwiseApplicabilityJudge(llm, self.settings)
            transfer_judge = MinimalTransferabilityJudge(llm, self.settings)
            reranker = (
                HttpReranker(
                    self.settings.rerank_base_url,
                    timeout_s=self.settings.rerank_timeout_s,
                    api_key=self.settings.rerank_api_key,
                    model=self.settings.rerank_model,
                )
                if self.settings.rerank_base_url and not self.settings.transferability_enabled
                else None
            )
            self.search_engine = SearchEngine(
                self.store, self.settings,
                judge=judge, reranker=reranker,
                card_vector_index=card_vector_index,
                transferability_judge=transfer_judge,
            )
        self.guard = get_guard(self.settings)
        self.counter = get_counter(self.settings)

    def _scope_lock(self, user_id):
        # Waiters retain strong references; unused scope locks can disappear.
        with self._scope_locks_guard:
            lock = self._scope_locks.get(user_id)
            if lock is None:
                lock = threading.RLock()
                self._scope_locks[user_id] = lock
            return lock

    # -- validation ------------------------------------------------------

    def validate_add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        raw_messages: object,
    ) -> list[MessageInput]:
        if not request_id or not isinstance(request_id, str):
            raise ValidationError("request_id is required and must be a string")
        if not user_id or not isinstance(user_id, str):
            raise ValidationError("user_id is required and must be a string")
        if not session_id or not isinstance(session_id, str):
            raise ValidationError("session_id is required and must be a string")
        if not isinstance(raw_messages, list):
            raise ValidationError("messages is required and must be an array")
        if not raw_messages:
            raise ValidationError("messages must not be empty")
        if len(raw_messages) > self.settings.max_messages_per_request:
            raise ValidationError(
                f"messages exceeds the configured limit of "
                f"{self.settings.max_messages_per_request} per request"
            )

        messages: list[MessageInput] = []
        total_chars = 0
        for index, item in enumerate(raw_messages):
            if not isinstance(item, dict):
                raise ValidationError(f"messages[{index}] must be an object")
            role = item.get("role")
            if role not in ("user", "assistant"):
                raise ValidationError(
                    f"messages[{index}].role must be 'user' or 'assistant', got {role!r}"
                )
            content = item.get("content")
            if content is None:
                raise ValidationError(f"messages[{index}].content is required")
            if not isinstance(content, str):
                # api_guide.md uses ContentPart[] only for the multimodal track.
                raise ValidationError(
                    f"messages[{index}].content must be a string for the code track"
                )
            if len(content) > self.settings.max_message_chars:
                raise ValidationError(
                    f"messages[{index}].content exceeds the configured limit of "
                    f"{self.settings.max_message_chars} characters"
                )
            timestamp = item.get("timestamp")
            if timestamp is not None and not isinstance(timestamp, int):
                raise ValidationError(
                    f"messages[{index}].timestamp must be an integer (Unix ms) when present"
                )
            total_chars += len(content)
            messages.append(MessageInput(role=role, content=content, timestamp=timestamp))

        if total_chars > self.settings.max_request_chars:
            raise ValidationError(
                f"total content exceeds the configured limit of "
                f"{self.settings.max_request_chars} characters"
            )
        return messages

    def validate_search(self, *, user_id: str, query: object, top_k: object, options: object) -> tuple[str, int, list[str]]:
        if not user_id or not isinstance(user_id, str):
            raise ValidationError("user_id is required and must be a string")
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("query is required and must be a non-empty string")
        if top_k is None:
            raise ValidationError("top_k is required")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValidationError("top_k must be an integer")
        if top_k < 0:
            raise ValidationError("top_k must not be negative")
        if top_k > self.settings.top_k_max:
            raise ValidationError(
                f"top_k exceeds the configured maximum of {self.settings.top_k_max}"
            )
        parsed_options: list[str] = []
        if options is not None:
            if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
                raise ValidationError("options must be an array of strings when present")
            parsed_options = list(options)
        return user_id, top_k, parsed_options

    # -- add -------------------------------------------------------------

    @_serialize_scope
    def add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        raw_messages: object,
    ) -> AddOutcome:
        started = time.monotonic()
        messages = self.validate_add(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            raw_messages=raw_messages,
        )

        # Decide the processing mode up front (plan.md A1/A4 + section 5 guard).
        # The guard's own sliding window decides; a request that arrives while
        # the system is degraded runs in the cheaper mode.
        mode = self.guard.plan_for_request(self.settings.add_sync_budget_s)

        budget = ModuleBudget(self.settings.add_sync_budget_s)
        digest = content_hash(messages)

        try:
            state = self.store.begin_ingestion(
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                digest=digest,
                mode=mode,
                lease_seconds=max(60.0, self.settings.add_sync_budget_s * 3),
            )
        except IngestionConflict as exc:
            raise ValidationError(str(exc)) from exc
        except IngestionInProgress as exc:
            raise IngestionInProgress(str(exc)) from exc

        if state.status == "committed":
            # P2.6 fix: Idempotent replay requires session_id consistency
            if state.session_id != session_id:
                raise ValidationError(
                    f"Request {request_id} already committed under session_id={state.session_id}; "
                    f"cannot replay with different session_id={session_id}"
                )
            # Idempotent replay: same key, same body, same session, already durable.
            completion = []
            if self.settings.add_finalize_sync:
                completion = self._finish_add_session(user_id, session_id, state.ingestion_id)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.guard.observe(
                elapsed_s=elapsed_ms / 1000.0,
                budget_s=self.settings.add_sync_budget_s,
                overran=False,
            )
            return AddOutcome(
                success=True,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                mode=state.mode,
                deduplicated=True,
                steps=[{"name": "idempotent_replay", "duration_ms": elapsed_ms, "skipped": True}] + completion,
                elapsed_ms=elapsed_ms,
            )

        try:
            # --- sync critical path (no model, no network) --------------
            chunks = budget.guard(
                "chunk",
                lambda: build_chunks(
                    user_id=user_id,
                    request_id=request_id,
                    messages=messages,
                    settings=self.settings,
                ),
            )
            if chunks is None:
                # Chunking is pure CPU and has no fallback; treat as failure
                # rather than returning a success we cannot honour.
                raise ValidationError("chunking did not complete within the request budget")

            sync_ms = int((time.monotonic() - started) * 1000)
            self.store.commit_chunks(
                state=state,
                messages=messages,
                chunks=chunks,
                sync_ms=sync_ms,
            )
        except ValidationError:
            self.store.mark_failed(state.ingestion_id)
            raise
        except Exception:
            self.store.mark_failed(state.ingestion_id)
            raise

        sync_complete_ms = int((time.monotonic() - started) * 1000)

        if self.settings.add_finalize_sync:
            completion = self._finish_add_session(user_id, session_id, state.ingestion_id)
            self.guard.observe(
                elapsed_s=sync_complete_ms / 1000.0,
                budget_s=self.settings.add_sync_budget_s,
                overran=sync_complete_ms / 1000.0 > self.settings.add_sync_budget_s,
            )
            return AddOutcome(
                success=True, request_id=request_id, user_id=user_id,
                session_id=session_id, mode=mode,
                steps=[{"name": s.name, "duration_ms": s.duration_ms, "skipped": s.skipped}
                       for s in budget.steps] + completion,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )

        # --- best-effort finalisation of closed sessions -----------------
        # This Add's session_id differs from earlier ones -> those sessions are
        # closed and get their cards now. Every failure is recorded rather than
        # raised -- a failed finalisation must not turn a successful Add into
        # an error; the session stays pending and Search retries it.
        enrichment_steps: list[dict] = []
        if mode == MODE_FULL and self.settings.experience_enabled:
            enrichment_steps.extend(
                self._finalize_closed_sessions(
                    user_id=user_id,
                    current_session_id=session_id,
                    trigger_ingestion_id=state.ingestion_id,
                )
            )
        else:
            enrichment_steps.append({
                "name": "finalize", "duration_ms": 0, "skipped": True,
                "reason": f"mode={mode}" if mode != MODE_FULL else "experience_disabled",
            })
        enrichment_steps.append(
            self._spawn_background_finalize(user_id, state.ingestion_id))

        total_ms = int((time.monotonic() - started) * 1000)

        # The guard protects the synchronous write path.  Finalisation has its
        # own deadline and must not make every session-boundary Add look like
        # a degraded service.
        self.guard.observe(
            elapsed_s=sync_complete_ms / 1000.0,
            budget_s=self.settings.add_sync_budget_s,
            overran=sync_complete_ms / 1000.0 > self.settings.add_sync_budget_s,
        )

        steps = [{"name": s.name, "duration_ms": s.duration_ms, "skipped": s.skipped} for s in budget.steps]
        steps.extend(enrichment_steps)

        return AddOutcome(
            success=True,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            mode=mode,
            steps=steps,
            elapsed_ms=total_ms,
        )

    # -- session finalisation ----------------------------------------------

    def _finish_add_session(self, user_id: str, session_id: str, ingestion_id: int) -> list[dict]:
        """Retry a committed snapshot without re-appending its chunks."""
        if self.store.session_is_extracted(user_id, session_id):
            return []
        if not self.settings.experience_enabled:
            raise LLMUnavailable("synchronous extraction is disabled")
        step = self._finalize_session(
            user_id=user_id, session_id=session_id,
            trigger_ingestion_id=ingestion_id,
            deadline=time.monotonic() + self.settings.add_enrich_budget_s,
        )
        if not self.store.session_is_extracted(user_id, session_id):
            # A durable chunk commit is not a successful synchronous Add yet.
            # Keep it for the platform's retry, but never serve its raw content.
            raise LLMUnavailable("session extraction did not complete")
        return [step]

    def _spawn_background_finalize(self, user_id: str, trigger_ingestion_id: int) -> dict:
        """Extract cards for the just-written session off the Add critical path.

        The Add contract requires ``success=true`` to imply immediate
        retrievability; card extraction is LLM-bound, so it must not run inside
        the Add response.  The worker finalises every pending session of the
        scope (the sync path above already closed the ones it had time for),
        bounded by the same enrichment budget it would have had inline.  The
        Search backstop stays as the fallback for anything this does not
        finish.  Failures leave the session pending and are retried there.
        """
        if self.settings.add_finalize_sync or not self.settings.finalize_background or not self.settings.experience_enabled:
            return {"name": "finalize_background", "duration_ms": 0, "skipped": True,
                    "reason": "disabled"}
        with self._bg_finalize_lock:
            if user_id in self._bg_finalize_users:
                # A worker for this scope is already running; its pending scan
                # plus the next Search's backstop cover the new session.
                return {"name": "finalize_background", "duration_ms": 0, "skipped": True,
                        "reason": "worker_in_flight"}
            self._bg_finalize_users.add(user_id)

        def _run() -> None:
            try:
                self._finalize_closed_sessions(
                    user_id=user_id,
                    current_session_id=None,
                    trigger_ingestion_id=trigger_ingestion_id,
                    budget_override=self.settings.add_enrich_budget_s,
                )
            except Exception:
                # Never surface background enrichment as an Add failure; the
                # session stays pending and the Search backstop retries it.
                pass
            finally:
                with self._bg_finalize_lock:
                    self._bg_finalize_users.discard(user_id)

        threading.Thread(target=_run, name=f"finalize:{user_id}",
                         daemon=True).start()
        return {"name": "finalize_background", "duration_ms": 0, "skipped": False}

    @_serialize_scope
    def _finalize_closed_sessions(
        self,
        *,
        user_id: str,
        current_session_id: str | None,
        trigger_ingestion_id: int,
        budget_override: float | None = None,
    ) -> list[dict]:
        """Extract cards for every closed session of this scope.

        ``current_session_id`` excludes the just-written session on the Add
        path (a session is only closed once a *different* session_id arrives);
        Search passes ``None`` so the still-open tail session is finalised too
        -- there is no other signal that would ever close it.
        """
        import time as _time

        if not self.settings.experience_enabled:
            return [{"name": "finalize", "duration_ms": 0, "skipped": True,
                     "reason": "experience_disabled"}]
        pending = self.store.pending_sessions(user_id, exclude_session_id=current_session_id)
        if not pending:
            return [{"name": "finalize", "duration_ms": 0, "skipped": True,
                     "reason": "no_closed_sessions"}]
        steps: list[dict] = []
        for session_id in pending:
            base_budget = self.settings.experience_budget_s
            if budget_override is not None:
                base_budget = budget_override
            deadline = _time.monotonic() + min(
                base_budget,
                self.settings.add_enrich_budget_s,
            )
            if deadline - _time.monotonic() <= 0.5:
                steps.append({"name": "finalize", "session": session_id,
                              "duration_ms": 0, "skipped": True,
                              "reason": "budget_exhausted"})
                continue
            steps.append(
                self._finalize_session(
                    user_id=user_id,
                    session_id=session_id,
                    trigger_ingestion_id=trigger_ingestion_id,
                    deadline=deadline,
                )
            )
        return steps

    @_serialize_scope
    def _finalize_session(
        self,
        *,
        user_id: str,
        session_id: str,
        trigger_ingestion_id: int,
        deadline: float,
    ) -> dict:
        """Extract, replace and index the cards of one closed session.

        Replacement is idempotent: a session reopened by a later Add goes back
        to pending and its next finalisation deletes these cards first. If the
        model calls fail outright the previous cards are kept and the session
        stays pending (Search will retry); a completed extraction that yields
        zero admissible cards still marks the session extracted, so a
        noise-only session is not re-extracted forever.
        """
        import time as _time

        started = _time.monotonic()
        step: dict = {"name": "finalize", "session": session_id}
        version = self.settings.experience_extraction_version
        existing = self.store.session_extraction_versions(user_id, session_id)
        # Cross-form replacement is unsafe: card-v4 packets and idea-level
        # cards carry different provenance contracts.  Keep any existing card
        # unless the session is re-extracted by the same version.
        if existing - {version}:
            return {**step, "duration_ms": 0, "skipped": True,
                    "reason": "extraction_version_conflict"}
        llm = self._llm or get_llm(self.settings)
        if not getattr(llm, "configured", False):
            return {**step, "duration_ms": 0, "skipped": True,
                    "reason": "llm_not_configured"}
        if not trigger_ingestion_id:
            # Search-triggered finalisation owns no ingestion of its own; the
            # session's latest Add stands in as the provenance row.
            trigger_ingestion_id = self.store.last_ingestion_of_session(
                user_id, session_id) or 0

        session_chunks = self.store.fetch_session_chunks(user_id, session_id, limit=-1)
        keep = [chunk for chunk in session_chunks if not is_setup_chunk(chunk.text)]
        if not keep:
            with self.store.write_tx():
                self.store.delete_session_experiences(user_id, session_id)
                self.store.mark_session_extracted(
                    user_id=user_id, session_id=session_id,
                    extraction_version=self.settings.experience_extraction_version,
                    card_count=0,
                )
            return {**step, "duration_ms": int((_time.monotonic() - started) * 1000),
                    "skipped": True, "reason": "setup_only", "cards": 0}

        call_budget = ModelCallBudget(self.settings.extract_calls_per_finalization)
        # The card-v4 setup filter protects a repair-packet window; the
        # idea-level extractor needs the closing summary even when its first
        # lines mention a clone/install.  It has its own noise filters.
        extract_keep = session_chunks if is_minimal(
            self.settings.experience_extraction_version) else keep
        cards, extraction_ok, extract_stats = self._extract_session_cards(
            user_id=user_id,
            session_id=session_id,
            keep=extract_keep,
            deadline=deadline,
            call_budget=call_budget,
        )
        step.update(extract_stats)

        if not extraction_ok:
            # The model never answered; keep any previous cards and stay
            # pending so a later trigger retries with the full timeline.
            return {**step, "duration_ms": int((_time.monotonic() - started) * 1000),
                    "skipped": True, "reason": extract_stats.get("reason") or "extraction_failed"}

        chunks_by_id = {chunk.chunk_id: chunk for chunk in extract_keep}
        if self.settings.add_finalize_sync:
            # Publish all card/index/status changes atomically. Network-bound
            # embedding happens after COMMIT, outside the SQLite writer lock.
            with self.store.write_tx():
                self.store.delete_session_experiences(user_id, session_id)
                stored_ids = self._store_extracted_cards(
                    cards, user_id=user_id, session_id=session_id,
                    trigger_ingestion_id=trigger_ingestion_id,
                    chunks_by_id=chunks_by_id, embed=False, strict=True,
                )
                self.store.mark_session_extracted(
                    user_id=user_id, session_id=session_id,
                    extraction_version=version, card_count=len(stored_ids),
                )
            if stored_ids:
                self._embed_cards(user_id, cards)
            return {**step, "duration_ms": int((_time.monotonic() - started) * 1000),
                    "skipped": not stored_ids, "cards": len(stored_ids)}

        self.store.delete_session_experiences(user_id, session_id)
        stored_ids = self._store_extracted_cards(
            cards,
            user_id=user_id,
            session_id=session_id,
            trigger_ingestion_id=trigger_ingestion_id,
            chunks_by_id=chunks_by_id,
        )
        self.store.mark_session_extracted(
            user_id=user_id, session_id=session_id,
            extraction_version=self.settings.experience_extraction_version,
            card_count=len(stored_ids),
        )
        step.update({
            "duration_ms": int((_time.monotonic() - started) * 1000),
            "skipped": not stored_ids,
            "cards": len(stored_ids),
        })
        return step

    def _extract_minimal_card(
        self,
        *,
        user_id: str,
        session_id: str,
        keep: list,
        deadline: float,
        call_budget: ModelCallBudget,
    ) -> tuple[list, bool, dict]:
        """One idea-level card from a curated issue + final-summary window.

        The full session timeline is not shown to the model: the final summary
        is the mechanism source, the first message is the retrieval face and
        test-shaped chunks are the acceptance-semantics source.  This keeps the
        call small, makes the noise/signal split explicit, and removes the
        historical action from the input before the model can copy it.
        """
        import time as _time

        stats = {"window": len(keep), "batches": 1, "calls": 0, "repairs": 0,
                 "rejected_cards": 0, "reason": None, "mode": "minimal"}
        evidence = build_minimal_evidence(
            keep, summary_fallback=self.settings.minimal_summary_fallback)
        if not evidence.is_usable():
            # No issue, or no closing summary and no fix diff to reconstruct
            # from: there is nothing to ground an idea-level card in.  Mark
            # the session extracted so a Search does not retry an incomplete
            # timeline forever.
            return [], True, {**stats, "reason": "no_issue_or_summary"}
        if evidence.is_fallback:
            stats["mode"] = "minimal_fallback"

        llm = self._llm or get_llm(self.settings)
        if not getattr(llm, "configured", False):
            return [], False, {**stats, "reason": "llm_not_configured"}

        chunks_by_id = {chunk.chunk_id: chunk for chunk in evidence.chunks}
        repair_notes: list[str] = []
        last_reason = "minimal_admission_failed"
        for attempt in range(2):
            remaining = deadline - _time.monotonic()
            if remaining <= 0.5:
                return [], False, {**stats, "reason": "budget_exhausted"}
            if not call_budget.take():
                return [], False, {**stats, "reason": "call_budget"}
            limiter = get_model_limiter(self.settings)
            wait_s = max(0.0, remaining - 0.5) if self.settings.add_finalize_sync else None
            if not limiter.acquire(timeout_s=wait_s):
                return [], False, {**stats, "reason": "model_concurrency"}
            stats["calls"] += 1
            try:
                remaining = deadline - _time.monotonic()
                if remaining <= 0.5:
                    return [], False, {**stats, "reason": "budget_exhausted"}
                user = build_minimal_prompt(evidence, 1)
                if repair_notes:
                    user += (
                        "\n上一轮输出的 card 未通过程序化准入校验，具体问题：\n"
                        + "\n".join(f"- {note}" for note in repair_notes)
                        + "\n请只修正这些问题并重新输出完整 JSON；仍无依据时输出 "
                          '{"cards":[]}。'
                    )
                call_options = {"max_retries": 0} if self.settings.add_finalize_sync else {}
                parsed, result = llm.complete_json(
                    system=extraction_system_prompt(MINIMAL_VERSION),
                    user=user,
                    max_tokens=self.settings.experience_max_output_tokens,
                    timeout_s=remaining,
                    **call_options,
                )
            except Exception:
                last_reason = "llm_error"
                continue
            finally:
                limiter.release()
            if getattr(result, "raw_finish_reason", None) == "length":
                last_reason = "truncated"
                continue
            if not isinstance(parsed, dict) or not isinstance(parsed.get("cards"), list):
                last_reason = "parse_error"
                continue
            proposed = parsed["cards"]
            if not proposed:
                # An explicit "no transferable experience" verdict is a valid
                # outcome and may replace previous cards.
                return [], True, {**stats, "reason": "no_experience"}
            stats["rejected_cards"] += len(proposed)
            cards = build_cards(
                raw=parsed,
                chunks_by_id=chunks_by_id,
                grounding_chunks_by_id=chunks_by_id,
                user_id=user_id,
                session_id=session_id,
                id_for=lambda i: experience_id_for(user_id, session_id, i),
                extraction_version=MINIMAL_VERSION,
                created_at=now_iso(),
                max_cards=1,
                mechanism_source="reconstructed" if evidence.is_fallback else "summary",
            )
            if cards:
                stats["rejected_cards"] -= 1
                return cards, True, stats
            last_reason = "minimal_admission_failed"
            repair_notes = card_admission_problems(
                proposed,
                chunks_by_id=chunks_by_id,
                user_id=user_id,
                session_id=session_id,
            )
            if attempt == 0:
                stats["repairs"] += 1
        return [], False, {**stats, "reason": last_reason}

    def _extract_session_cards(
        self,
        *,
        user_id: str,
        session_id: str,
        keep: list,
        deadline: float,
        call_budget: ModelCallBudget,
    ) -> tuple[list, bool, dict]:
        """Run the extraction LLM over the session timeline, in bounded batches.

        Chunks are read as one timeline across every Add of the session, so a
        problem solved across several Adds is extracted as one case. A session
        larger than one batch is split into several calls, bounded by the
        per-finalisation call budget.
        """
        import time as _time

        if is_minimal(self.settings.experience_extraction_version):
            return self._extract_minimal_card(
                user_id=user_id, session_id=session_id, keep=keep,
                deadline=deadline, call_budget=call_budget,
            )

        llm = self._llm or get_llm(self.settings)
        batch_size = max(1, self.settings.experience_session_max_chunks)
        max_calls = max(1, call_budget.limit)
        batches = [keep[i : i + batch_size] for i in range(0, len(keep), batch_size)]
        # The issue statement lives in the session's first chunks (batch 1).
        # A batch-2+ window without it cannot anchor a card's state to the
        # original symptoms, so every later window prepends a small problem
        # digest. Its chunks are first-class, citable evidence and join the
        # admission universe (no synthetic text is injected).
        prefix: list = []
        prefix_chars = 0
        for chunk in keep[:3]:
            prefix.append(chunk)
            prefix_chars += len(chunk.text)
            if prefix_chars >= 1200:
                break
        truncated = len(batches) > max_calls
        batches = batches[:max_calls]
        cards: list = []
        seen: set[tuple] = set()
        calls = 0
        retries = 0
        truncations = 0
        reason = None
        extraction_ok = False
        card_index = 0

        def reserve_ids(n: int) -> int:
            """Reserve a contiguous experience-id block across batches."""
            nonlocal card_index
            base = card_index
            card_index += max(1, n)
            return base

        def id_block_for(base: int):
            return lambda i, _b=base: experience_id_for(user_id, session_id, _b + i)

        def call_model(batch, card_hint: int | None = None):
            """One extraction call: (parsed, result, failure_reason)."""
            remaining = deadline - _time.monotonic()
            if remaining <= 0.5:
                return None, None, "budget_exhausted"
            limiter = get_model_limiter(self.settings)
            if not limiter.acquire():
                return None, None, "model_concurrency"
            try:
                user = build_extraction_prompt(
                    batch, self.settings.experience_quote_max_chars, multi_case=True,
                    max_cards=self.settings.experience_max_cards_per_session,
                    extraction_version=self.settings.experience_extraction_version,
                )
                if card_hint:
                    user += (
                        f"\n注意：本次最多输出 {card_hint} 张 card，"
                        "优先保证 JSON 完整；放不下的内容留空，不要续写。"
                    )
                parsed, result = llm.complete_json(
                    system=extraction_system_prompt(self.settings.experience_extraction_version),
                    user=user,
                    max_tokens=self.settings.experience_max_output_tokens,
                    timeout_s=remaining,
                )
                return parsed, result, None
            except Exception:
                return None, None, "llm_error"
            finally:
                limiter.release()

        for batch_no, batch in enumerate(batches):
            if not call_budget.take():
                reason = "call_budget"
                break
            window = batch if batch_no == 0 else prefix + batch
            calls += 1
            raw, result, failure = call_model(window)
            cut = getattr(result, "raw_finish_reason", None) == "length"
            if cut:
                truncations += 1
            # A completion cut at the token ceiling loses the whole batch: the
            # JSON never parses, so the session silently contributes nothing.
            # One cheaper retry (fewer cards, same evidence) is better than a
            # hole in the library, and it stays inside the budget.
            if (
                (not isinstance(raw, dict) or cut)
                and deadline - _time.monotonic() > 1.0
                and call_budget.take()
            ):
                calls += 1
                retries += 1
                raw, result, failure = call_model(window, 2)
                cut = getattr(result, "raw_finish_reason", None) == "length"
                if cut:
                    truncations += 1
            if failure:
                reason = failure
                continue
            if not isinstance(raw, dict):
                reason = "truncated" if cut else "parse_error"
                continue
            if cut:
                reason = "truncated"
                continue
            extraction_ok = True
            batch_by_id = {chunk.chunk_id: chunk for chunk in window}
            max_cards = self.settings.experience_max_cards_per_session
            new_cards = build_cards(
                raw=raw,
                chunks_by_id=batch_by_id,
                grounding_chunks_by_id={chunk.chunk_id: chunk for chunk in keep},
                user_id=user_id,
                session_id=session_id,
                id_for=id_block_for(reserve_ids(max_cards)),
                extraction_version=self.settings.experience_extraction_version,
                created_at=now_iso(),
                max_cards=max_cards,
            )
            for card in new_cards:
                key = (card.problem or "", tuple(card.files))
                if key in seen:
                    continue
                seen.add(key)
                cards.append(card)

        if len(cards) > self.settings.experience_max_cards_per_session:
            cards = cards[: self.settings.experience_max_cards_per_session]

        stats = {
            "window": len(keep),
            "batches": len(batches),
            "calls": calls,
            "retries": retries,
            "truncations": truncations,
            "truncated": truncated,
            "reason": reason,
        }
        return cards, extraction_ok, stats

    def _store_extracted_cards(
        self,
        cards,
        *,
        user_id: str,
        session_id: str,
        trigger_ingestion_id: int,
        chunks_by_id: dict,
        embed: bool = True,
        strict: bool = False,
    ) -> list[str]:
        if not cards:
            return []
        default_cap = self.settings.experience_tiers[0].per_item_tokens
        evidence = self.store.fetch_evidence_chunks(user_id, session_id)
        render_chunks = {chunk.chunk_id: chunk for chunk in evidence}
        render_chunks.update(chunks_by_id)
        enrich_pool = list(render_chunks.values())
        final_files = _final_diff_files(enrich_pool)
        stored: list[str] = []
        written = []
        for card in cards:
            try:
                if card.is_minimal:
                    # No deterministic packet: the three idea-level faces are
                    # the complete card.
                    full_content = render_minimal(card, self.counter, None)
                else:
                    card = enrich_card(card, enrich_pool)
                    card = _clamp_files_to_final_diff(card, final_files)
                    # Persist complete card content even if a later query has
                    # a smaller per-item budget. Retrieval may omit, never
                    # amputate.
                    full_content = render_experience(
                        card, render_chunks, self.counter, default_cap,
                        self.settings.experience_quote_max_chars,
                    )
                if not full_content.strip():
                    continue
                self.store.insert_experience(
                    user_id=user_id,
                    session_id=session_id,
                    experience_id=card.experience_id,
                    ingestion_id=trigger_ingestion_id,
                    kind=card.kind,
                    payload_json=json.dumps(card_to_payload(card), ensure_ascii=False),
                    full_content=full_content,
                    extraction_version=card.extraction_version,
                    created_at=card.created_at or now_iso(),
                    terms=card_field_terms(card),
                    entities=card_entities(card, render_chunks),
                    anchors=card_anchors(card),
                )
                stored.append(card.experience_id)
                written.append(card)
            except Exception:
                if strict:
                    raise
                continue
        if written and embed:
            self._embed_cards(user_id, written)
        return stored

    def _embed_cards(self, user_id: str, cards) -> int:
        """Best-effort card vectors, written after the card commit.

        A card is the retrieval unit, so it carries an embedding of its own
        index face (``card_retrieval_text``). Failure returns 0 and leaves the
        card fully searchable through its lexical channels -- enrichment may
        only add retrievable memory, never retract a confirmed Add.
        """
        embedder = self.embedder
        if embedder is None or not embedder.available:
            return 0
        budget = min(self.settings.embed_budget_s, self.settings.add_enrich_budget_s)
        texts = [card_retrieval_text(card) for card in cards]
        try:
            result = embedder.embed(texts, timeout_s=budget)
        except Exception:
            return 0
        if result is None or len(result.vectors) != len(cards):
            return 0
        rows = [
            (
                card.experience_id,
                self.settings.embed_model,
                result.dim,
                pack_vector(vector),
            )
            for card, vector in zip(cards, result.vectors)
        ]
        try:
            return self.store.store_experience_vectors(user_id, rows)
        except Exception:
            return 0

    @_serialize_scope
    def search(
        self,
        *,
        user_id: str,
        query: object,
        top_k: object,
        options: object = None,
    ) -> SearchOutcome:
        started = time.monotonic()
        user_id, top_k, parsed_options = self.validate_search(
            user_id=user_id, query=query, top_k=top_k, options=options
        )
        top_k = min(top_k, self.settings.search_max_items)

        budget = ModuleBudget(self.settings.search_hard_budget_s)
        data: list[dict] = []
        diagnostics: dict = {}

        if top_k == 0:
            return SearchOutcome(
                data=[],
                diagnostics={"selected": 0, "elapsed_ms": 0, "degraded": ["top_k_zero"]},
            )

        # Backstop: finalise every still-pending session of this scope. The
        # last session of an Add stream never sees a session_id change, so
        # without this its chunks would never become cards. The Add path also
        # leaves sessions pending when its own deadline ran out; this retries
        # them. Bounded so it cannot consume the whole search budget.
        finalized = [] if self.settings.add_finalize_sync else self._finalize_closed_sessions(
            user_id=user_id,
            current_session_id=None,
            trigger_ingestion_id=0,
            budget_override=self.settings.search_backstop_budget_s,
        )
        finalized_sessions = [
            step for step in finalized
            if not step.get("skipped") and step.get("name") == "finalize"
        ]

        # Storage failure must surface as an error, never as "no results"
        # (plan.md S6).
        # P1.1 fix: pass a real deadline so the engine can skip the optional
        # vector channel when time is short, rather than recording overruns
        # after the fact.
        data, diag = self.search_engine.search(
            user_id=user_id,
            query=query,
            top_k=top_k,
            options=parsed_options,
            deadline=budget,
        )

        pending_items = self._pending_session_items(
            user_id, query, parsed_options,
            slots=top_k - len(data), have_cards=bool(data))
        if pending_items:
            data = data + pending_items

        elapsed_ms = int((time.monotonic() - started) * 1000)
        diag.elapsed_ms = elapsed_ms

        diagnostics = {
            "channels": diag.channels,
            "pool_size": diag.pool_size,
            "after_dedup": diag.after_dedup,
            "selected": len(data),
            "finalized_sessions": len(finalized_sessions),
            "pending_fallback": len(pending_items),
            "dropped_duplicate": diag.dropped_duplicate,
            "dropped_budget": diag.dropped_budget,
            "dropped_final": diag.dropped_final,
            "patch_suppressed": diag.patch_suppressed,
            "dropped_low_score": diag.dropped_low_score,
            "tokens_used": diag.tokens_used,
            "token_mode": diag.token_mode,
            "degraded": diag.degraded,
            "applicability": diag.applicability,
            "rerank": diag.rerank,
            "elapsed_ms": elapsed_ms,
            "budget_exceeded": elapsed_ms > self.settings.search_budget_p95_s * 1000,
        }
        return SearchOutcome(data=data, diagnostics=diagnostics)

    def _pending_session_items(
        self,
        user_id: str,
        query: str,
        options: object,
        *,
        slots: int,
        have_cards: bool,
    ) -> list[dict]:
        """Provisional raw-chunk items for sessions that have no card yet.

        api_guide.md: Add's ``success=true`` means "persisted and immediately
        retrievable".  Card extraction is LLM-bound and asynchronous, so a
        Search issued right after Add can face sessions with no card.  This
        fallback surfaces their raw chunks as one provisional item per session,
        scored by lexical overlap with the query; once the session is
        finalised it leaves ``pending`` and its card replaces this item.
        Cards always take precedence: the items only fill slots the card
        ranking left free, and a zero-overlap session is dropped unless there
        are no card results at all.
        """
        if self.settings.add_finalize_sync or not self.settings.search_pending_fallback or slots <= 0:
            return []
        limit = min(self.settings.pending_fallback_max_sessions, slots)
        pending = self.store.pending_sessions(user_id)[:limit]
        if not pending:
            return []
        option_text = ""
        if isinstance(options, list):
            option_text = " ".join(str(o) for o in options if isinstance(o, str))
        query_words = set(_content_words(f"{query} {option_text}"))
        items: list[dict] = []
        for session_id in pending:
            chunks = self.store.fetch_session_chunks(user_id, session_id)
            parts: list[str] = []
            used = 0
            for chunk in chunks:
                text = chunk.text.strip()
                if not text:
                    continue
                if used + len(text) > self.settings.pending_fallback_max_chars:
                    break
                parts.append(text)
                used += len(text) + 2
            if not parts:
                continue
            content = "\n\n".join(parts)
            lowered = content.lower()
            matched = sum(1 for word in query_words if word in lowered)
            ratio = round(matched / len(query_words), 4) if query_words else 0.0
            if ratio <= 0.0 and have_cards:
                # Cards answered the query; raw noise adds nothing.
                continue
            items.append({
                "id": f"pending:{session_id}",
                "content": content,
                "score": ratio,
            })
        items.sort(key=lambda item: item["score"], reverse=True)
        return items[:slots]
