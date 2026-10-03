"""Add pipeline orchestration.

Order matters and encodes the third design invariant:

1. validate + idempotency check;
2. chunk (deterministic) and extract identifiers (deterministic);
3. persist everything and commit.

Because step 3 precedes any enrichment, a later LLM timeout cannot turn a valid
write into a failure — the raw memory is already durable and searchable, so we
still answer ``success: true``. Enrichment (L3 cards/episodes) is a separate
pass that can run incrementally without ever blocking the contract response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.config import Settings
from ..core.errors import PayloadTooLarge
from ..core.logging import get_logger
from ..index.store import AddOutcome, Store, sha256_text, utc_now_iso
from .chunker import chunk_content
from .entities import extract_entities

log = get_logger("codemem.add")


@dataclass(slots=True)
class AddPlan:
    messages: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    entities: dict[int, list[tuple[str, str, str]]]
    duplicate: bool = False


class AddPipeline:
    def __init__(self, settings: Settings, store: Store, embedder=None) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder

    # ------------------------------------------------------------ public --

    def handle(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: list[Any],
    ) -> tuple[AddOutcome, bool]:
        """Run Add. Returns ``(outcome, degraded)``.

        ``degraded`` is True when enrichment was skipped or failed; the response
        is still a success because the memory is durably searchable either way.
        """
        t0 = time.monotonic()

        total_chars = sum(len(m.content) for m in messages)
        if total_chars > self.settings.max_content_chars:
            raise PayloadTooLarge(
                f"content exceeds {self.settings.max_content_chars} characters",
                extra={"received_chars": total_chars},
            )

        plan = self._plan(request_id, user_id, session_id, messages)
        if plan.duplicate:
            return AddOutcome(0, 0, duplicate=True), False

        outcome = self.store.add_messages(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            messages=plan.messages,
            chunks=plan.chunks,
            entities=plan.entities,
            t0=t0,
        )

        # Embedding happens after the transaction commits, so the contract
        # response never depends on the encoder being available. A failure here
        # only means this memory is not dense-retrievable; the lexical and
        # identifier channels still reach it.
        if not outcome.duplicate:
            self._embed(user_id, request_id, plan, t0)

        degraded = True
        if self.settings.card_enabled or self.settings.llm_enabled:
            degraded = not self._enrich(user_id, session_id, plan, t0)
        return outcome, degraded

    def _embed(
        self, user_id: str, request_id: str, plan: AddPlan, t0: float
    ) -> None:
        """Embed this request's memories, if the dense channel is enabled."""
        instance = self.embedder
        if instance is None or not instance.available:
            return
        try:
            with self.store._read() as conn:  # noqa: SLF001 - same package
                rows = conn.execute(
                    "SELECT id, text FROM memory WHERE user_id = ? AND request_id = ?"
                    " ORDER BY ord",
                    (user_id, request_id),
                ).fetchall()
            rows = rows[: self.settings.dense_max_per_add]
            if not rows:
                return
            vectors = instance.embed([row["text"] for row in rows])
            if not vectors:
                return
            self.store.store_vectors(
                user_id, [(int(r["id"]), v) for r, v in zip(rows, vectors)]
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "embedding skipped",
                extra={"ctx": {"user_id": user_id, "error": str(exc)[:200]}},
            )

    # ------------------------------------------------------------ planning --

    def _plan(
        self, request_id: str, user_id: str, session_id: str, messages: list[Any]
    ) -> AddPlan:
        raw: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        entities: dict[int, list[tuple[str, str, str]]] = {}
        ord_counter = 0

        for msg_index, message in enumerate(messages):
            content = message.content
            raw.append(
                {
                    "msg_index": msg_index,
                    "role": message.role,
                    "ts": message.timestamp,
                    "content": content,
                }
            )

            pieces = chunk_content(
                content,
                target_tokens=self.settings.target_chunk_tokens,
                max_tokens=self.settings.max_chunk_tokens,
                hard_chars=self.settings.hard_chunk_chars,
            )
            for part_index, piece in enumerate(pieces):
                text = piece.text.strip()
                if not text:
                    continue
                ord_value = ord_counter
                ord_counter += 1
                chunks.append(
                    {
                        "msg_index": msg_index,
                        "part_index": part_index,
                        "part_count": piece.part_count,
                        "kind": piece.kind,
                        "lang": piece.lang,
                        # The memory title carries the structural kind so scoring
                        # can align a chunk with the question's intent without
                        # another join.
                        "title": f"{piece.kind}|{piece.lang or ''}",
                        "text": text,
                        "sha": sha256_text(text),
                        "line_start": piece.line_start,
                        "line_end": piece.line_end,
                        "ts": message.timestamp,
                        "ord": ord_value,
                    }
                )
                found = extract_entities(text, kind=piece.kind, lang=piece.lang)
                if found:
                    entities[ord_value] = [
                        (e.etype, e.value_norm, e.value_raw) for e in found
                    ]

        return AddPlan(messages=raw, chunks=chunks, entities=entities)

    # ---------------------------------------------------------- enrichment --

    _CARD_SYSTEM = (
        "You summarize software-engineering sessions for a retrieval index. "
        "Write concrete, factual prose; never invent files, commands, or outcomes "
        "that the transcript does not show."
    )

    _CARD_PROMPT = """Below is one recorded engineering session (user and assistant turns).

Transcript:
{transcript}

Summarize what this session did in at most 120 words: the problem being solved,
the approach taken, what was changed (concrete file paths and identifiers where
the transcript shows them), and the outcome. Output only the summary text."""

    def _enrich(
        self, user_id: str, session_id: str, plan: AddPlan, t0: float
    ) -> bool:
        """L3 enrichment: one experience card per session.

        The card is a session-level comparable object for ranking — the
        cross-encoder reads (query, overview) as one pair, which no single
        chunk of a ~100-chunk session can stand in for (the session head is
        the decisive entry only 12.9 % of the time). It is also a recall
        candidate: a session whose chunks share vocabulary with each other but
        not with the query can still be surfaced through its overview.

        Invariants (docs/DESIGN.md): the card never carries returned text —
        Search may score it but the assembler never emits it; identity and
        content are pure functions of Add input (the overview is cached by the
        prompt's content hash, so a re-Add reproduces the same row); and a
        failure here degrades Add instead of failing it.
        """
        if not self.settings.card_enabled:
            # No card work requested; nothing was skipped, so not degraded.
            return self.settings.llm_enabled

        deadline = t0 + self.settings.add_deadline_seconds
        try:
            overview = self._session_overview(plan, deadline)
            if not overview:
                return False

            ts_values = [m["ts"] for m in plan.messages if m.get("ts") is not None]
            ts = min(ts_values) if ts_values else None
            ord_value = (plan.chunks[-1]["ord"] + 1) if plan.chunks else 0
            found = extract_entities(overview)
            entities = [(e.etype, e.value_norm, e.value_raw) for e in found]

            memory_id = self.store.add_card(
                user_id=user_id,
                session_id=session_id,
                text=overview,
                ts=ts,
                ord=ord_value,
                entities=entities,
            )
            if memory_id is None:
                return True  # already stored by a previous identical Add

            self._embed_card(user_id, memory_id, overview)
            log.info(
                "card written",
                extra={"ctx": {"user_id": user_id, "session_id": session_id,
                               "memory_id": memory_id, "chars": len(overview)}},
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "enrichment failed; raw memory retained",
                extra={"ctx": {"user_id": user_id, "session_id": session_id,
                               "error": str(exc)[:200]}},
            )
            return False

    def _session_overview(self, plan: AddPlan, deadline: float) -> str | None:
        """The session's overview text, from cache or the LLM. Never raises."""
        budget = self.settings.card_max_input_chars
        turns = [
            f"[{m['role']}] {m['content']}" for m in plan.messages
        ]
        transcript = "\n\n".join(turns)
        if len(transcript) > budget:
            # Trajectories read before they change: keep the issue statement
            # from the head and the edits from the tail, drop the middle.
            head = budget // 2
            transcript = transcript[:head] + "\n\n[...]\n\n" + transcript[-head:]

        cache_key = sha256_text(self._CARD_PROMPT.format(transcript=transcript))
        with self.store._read() as conn:  # noqa: SLF001 - same package
            row = conn.execute(
                "SELECT payload FROM llm_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is not None:
            return row["payload"]

        if time.monotonic() >= deadline:
            log.warning("enrichment skipped: deadline reached")
            return None

        overview = self._llm_complete(
            self._CARD_SYSTEM,
            self._CARD_PROMPT.format(transcript=transcript),
        )
        if not overview:
            return None
        overview = overview.strip()
        if not overview:
            return None
        with self.store._write() as conn:  # noqa: SLF001 - same package
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache(cache_key, kind, payload, created_at)"
                " VALUES (?,?,?,?)",
                (cache_key, "card", overview, utc_now_iso()),
            )
        return overview

    def _llm_complete(self, system: str, user: str) -> str | None:
        """One chat completion against the configured relay. Never raises."""
        base_url = self.settings.llm_base_url
        api_key = self.settings.llm_api_key
        if not base_url or not api_key:
            log.warning("card skipped: LLM not configured")
            return None
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.settings.llm_timeout_seconds,
            )
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=self.settings.card_max_output_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "card LLM call failed",
                extra={"ctx": {"error": str(exc)[:200]}},
            )
            return None

    def _embed_card(self, user_id: str, memory_id: int, overview: str) -> None:
        """Give the card a dense vector so the paraphrase channel reaches it."""
        instance = self.embedder
        if instance is None or not instance.available:
            return
        try:
            vectors = instance.embed([overview])
            if vectors:
                self.store.store_vectors(user_id, [(memory_id, vectors[0])])
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "card embedding skipped",
                extra={"ctx": {"user_id": user_id, "error": str(exc)[:200]}},
            )
