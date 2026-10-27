"""Add / Search orchestration.

Add flow:

    validate -> idempotency -> reserve key -> chunk (CPU) -> COMMIT (chunks)
    -> finalise closed sessions (best-effort, LLM) -> return success

The ordering matters: the idempotency key is reserved *before* any expensive
work so concurrent duplicates cannot both proceed, and the success response is
sent only after the chunk transaction has committed, so ``success: true``
always means "persisted" as api_guide.md requires.

Cards are extracted per **session**, not per Add. A session is considered
closed the moment an Add with a *different* session_id arrives (``session_id``
change is the only end-of-session signal the API offers); Search finalises
whatever is still pending as a backstop, so the last session of a scope also
gets its cards. Chunks are the persisted minimal facts -- raw messages are
never stored, and only cards are embedded.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from uuid import uuid4

from .common.chunking import build_chunks
from .add.attempt import is_v5, render_attempt
from .common.config import Settings, get_settings
from .common.embeddings import get_embedder, get_model_limiter, pack_vector
from .search.applicability import ListwiseApplicabilityJudge
from .add.experience import (
    extraction_system_prompt,
    build_cards,
    build_extraction_prompt,
    card_admission_problems,
    card_anchors,
    card_entities,
    card_field_terms,
    card_from_payload,
    card_retrieval_text,
    card_to_payload,
    enrich_card,
    is_setup_chunk,
    render_experience,
)
from .common.guard import MODE_FULL, ModuleBudget, get_guard
from .common.llm import get_llm
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

_FINAL_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+)", re.M)


def _final_diff_files(chunks) -> frozenset:
    """Files of the session's final staged diff, scanned from chunk texts.

    300Lite-style sessions end with ``git diff --cached``; the staged diff is
    the mechanical ground truth of what survived.  Returns an empty set when
    no such chunk exists (the clamp then no-ops -- soft by design, the final
    diff is a harness convention, not a guarantee).
    """
    import re as _re

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
    if card.is_attempt or not final_files or not card.files:
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
            judge = ListwiseApplicabilityJudge(
                self._llm or get_llm(self.settings), self.settings
            )
            reranker = (
                HttpReranker(
                    self.settings.rerank_base_url,
                    timeout_s=self.settings.rerank_timeout_s,
                    api_key=self.settings.rerank_api_key,
                    model=self.settings.rerank_model,
                )
                if self.settings.rerank_base_url
                else None
            )
            self.search_engine = SearchEngine(
                self.store, self.settings,
                judge=judge, reranker=reranker,
                card_vector_index=card_vector_index,
            )
        self.guard = get_guard(self.settings)
        self.counter = get_counter(self.settings)

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
                steps=[{"name": "idempotent_replay", "duration_ms": elapsed_ms, "skipped": True}],
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

    def _finalize_closed_sessions(
        self,
        *,
        user_id: str,
        current_session_id: str | None,
        trigger_ingestion_id: int,
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
            deadline = _time.monotonic() + min(
                self.settings.experience_budget_s,
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
        if existing - {version} and (is_v5(version) or any(is_v5(v) for v in existing)):
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
            self.store.delete_session_experiences(user_id, session_id)
            self.store.mark_session_extracted(
                user_id=user_id, session_id=session_id,
                extraction_version=self.settings.experience_extraction_version,
                card_count=0,
            )
            return {**step, "duration_ms": int((_time.monotonic() - started) * 1000),
                    "skipped": True, "reason": "setup_only", "cards": 0}

        call_budget = ModelCallBudget(self.settings.extract_calls_per_finalization)
        cards, extraction_ok, extract_stats = self._extract_session_cards(
            user_id=user_id,
            session_id=session_id,
            keep=keep,
            deadline=deadline,
            call_budget=call_budget,
        )
        step.update(extract_stats)

        if not extraction_ok:
            # The model never answered; keep any previous cards and stay
            # pending so a later trigger retries with the full timeline.
            return {**step, "duration_ms": int((_time.monotonic() - started) * 1000),
                    "skipped": True, "reason": extract_stats.get("reason") or "extraction_failed"}

        self.store.delete_session_experiences(user_id, session_id)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in keep}
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

        llm = self._llm or get_llm(self.settings)
        batch_size = max(1, self.settings.experience_session_max_chunks)
        max_calls = max(1, call_budget.limit)
        batches = [keep[i : i + batch_size] for i in range(0, len(keep), batch_size)]
        truncated = len(batches) > max_calls
        batches = batches[:max_calls]
        cards: list = []
        seen: set[tuple] = set()
        calls = 0
        retries = 0
        repairs = 0
        truncations = 0
        rejected_cards = 0
        reason = None
        extraction_ok = False
        card_index = 0

        def id_for(_local: int) -> str:
            nonlocal card_index
            experience_id = experience_id_for(user_id, session_id, card_index)
            card_index += 1
            return experience_id

        def call_model(batch, card_hint: int | None, repair_notes: list[str] | None = None):
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
                if repair_notes:
                    user += (
                        "\n你上一轮输出的全部 card 均未通过程序化准入校验，具体问题：\n"
                        + "\n".join(f"- {note}" for note in repair_notes)
                        + "\n请在同一输入上修正后重新输出完整 JSON："
                        "state 证据取自动作之前的行（issue 描述通常在最前面），"
                        "result 证据取动作之后含实际输出的行，"
                        "行范围不超过 800 字符，不要引用空行或纯标记行。"
                        "宁可少出或不出卡，不要为了通过校验改出更简单或更外围的卡："
                        "clone/checkout、搭建环境、todo/进度记录类动作仍不合格，"
                        "卡必须针对问题本身的检查或修改。"
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

        for batch in batches:
            if not call_budget.take():
                reason = "call_budget"
                break
            calls += 1
            raw, result, failure = call_model(batch, None)
            cut = getattr(result, "raw_finish_reason", None) == "length"
            if cut:
                truncations += 1
            # A completion cut at the token ceiling loses the whole batch: the
            # JSON never parses, so the session silently contributes nothing.
            # One cheaper retry (fewer cards, same evidence) is much better
            # than a hole in the library, and it stays inside the budget.
            if (
                (not isinstance(raw, dict) or cut)
                and deadline - _time.monotonic() > 1.0
                and call_budget.take()
            ):
                calls += 1
                retries += 1
                raw, result, failure = call_model(batch, 2)
                cut = getattr(result, "raw_finish_reason", None) == "length"
                if cut:
                    truncations += 1
            if isinstance(raw, dict) and not is_v5(self.settings.experience_extraction_version):
                extraction_ok = True
            if failure:
                reason = failure
                continue
            if not isinstance(raw, dict):
                reason = "truncated" if cut else "parse_error"
                continue
            if cut and is_v5(self.settings.experience_extraction_version):
                reason = "truncated"
                continue
            batch_by_id = {chunk.chunk_id: chunk for chunk in batch}
            max_cards = self.settings.experience_max_cards_per_session
            new_cards = build_cards(
                raw=raw,
                chunks_by_id=batch_by_id,
                grounding_chunks_by_id={chunk.chunk_id: chunk for chunk in keep},
                user_id=user_id,
                session_id=session_id,
                id_for=id_for,
                extraction_version=self.settings.experience_extraction_version,
                created_at=now_iso(),
                max_cards=max_cards,
            )
            if is_v5(self.settings.experience_extraction_version):
                proposed = raw.get("cards")
                if isinstance(proposed, list):
                    rejected_cards += max(0, len(proposed) - len(new_cards))
                # Union sampling: per-attempt success is moderate and the
                # failures are independent dice rolls, so K independent
                # samples over the same window recover recall. The admission
                # gate is untouched — only cards that pass get merged — and
                # the per-session card cap bounds the union.
                saw_empty_verdict = isinstance(proposed, list) and not proposed
                saw_proposals = isinstance(proposed, list) and bool(proposed)
                for _ in range(max(0, self.settings.experience_v5_sample_calls - 1)):
                    if (not call_budget.take()
                            or deadline - _time.monotonic() <= 1.0):
                        break
                    calls += 1
                    raw_s, result_s, failure_s = call_model(batch, None)
                    cut_s = getattr(result_s, "raw_finish_reason", None) == "length"
                    if cut_s:
                        truncations += 1
                    if failure_s or not isinstance(raw_s, dict) or cut_s:
                        continue
                    proposed_s = raw_s.get("cards")
                    cards_s = build_cards(
                        raw=raw_s,
                        chunks_by_id=batch_by_id,
                        grounding_chunks_by_id={chunk.chunk_id: chunk for chunk in keep},
                        user_id=user_id,
                        session_id=session_id,
                        id_for=id_for,
                        extraction_version=self.settings.experience_extraction_version,
                        created_at=now_iso(),
                        max_cards=max_cards,
                    )
                    if isinstance(proposed_s, list):
                        rejected_cards += max(0, len(proposed_s) - len(cards_s))
                        saw_proposals = saw_proposals or bool(proposed_s)
                        saw_empty_verdict = saw_empty_verdict or not proposed_s
                    new_cards = (new_cards + cards_s)[:max_cards]
                # Feedback repair: when proposals exist but none survived
                # admission, show the labeled rejection reasons over the same
                # window (line numbers stay valid) and let the model
                # re-select. At most two rounds, inside the call budget.
                repair_rounds = 0
                while (saw_proposals and not new_cards
                       and repair_rounds < 2
                       and deadline - _time.monotonic() > 1.0
                       and call_budget.take()):
                    repair_rounds += 1
                    calls += 1
                    repairs += 1
                    problems = card_admission_problems(
                        proposed or [], chunks_by_id=batch_by_id,
                        user_id=user_id, session_id=session_id)
                    raw, result, failure = call_model(batch, None, repair_notes=problems)
                    if not isinstance(raw, dict):
                        continue
                    proposed = raw.get("cards")
                    cards_r = build_cards(
                        raw=raw,
                        chunks_by_id=batch_by_id,
                        grounding_chunks_by_id={chunk.chunk_id: chunk for chunk in keep},
                        user_id=user_id,
                        session_id=session_id,
                        id_for=id_for,
                        extraction_version=self.settings.experience_extraction_version,
                        created_at=now_iso(),
                        max_cards=max_cards,
                    )
                    if isinstance(proposed, list):
                        rejected_cards += max(0, len(proposed) - len(cards_r))
                        saw_proposals = saw_proposals or bool(proposed)
                        saw_empty_verdict = saw_empty_verdict or not proposed
                    new_cards = (new_cards + cards_r)[:max_cards]
                if new_cards or (not saw_proposals and saw_empty_verdict):
                    extraction_ok = True
                else:
                    # Malformed/ungrounded nonempty output is not a genuine
                    # "no experience" verdict; preserve existing cards.
                    reason = "attempt_admission_failed"
            for card in new_cards:
                key = ((card.state, card.action, card.result) if card.is_attempt
                       else (card.problem or "", tuple(card.files)))
                if key in seen:
                    continue
                seen.add(key)
                cards.append(card)

        stats = {
            "window": len(keep),
            "batches": len(batches),
            "calls": calls,
            "retries": retries,
            "repairs": repairs,
            "truncations": truncations,
            "rejected_cards": rejected_cards,
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
                card = enrich_card(card, enrich_pool)
                card = _clamp_files_to_final_diff(card, final_files)
                # Persist complete v5 content even if a later query has a
                # smaller per-item budget. Retrieval may omit, never amputate.
                if card.is_attempt:
                    full_content = render_attempt(card, self.counter, None)
                else:
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
                continue
        if written:
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
        finalized = self._finalize_closed_sessions(
            user_id=user_id,
            current_session_id=None,
            trigger_ingestion_id=0,
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
        elapsed_ms = int((time.monotonic() - started) * 1000)
        diag.elapsed_ms = elapsed_ms

        diagnostics = {
            "channels": diag.channels,
            "pool_size": diag.pool_size,
            "after_dedup": diag.after_dedup,
            "selected": len(data),
            "finalized_sessions": len(finalized_sessions),
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
