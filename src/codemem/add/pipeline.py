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
from ..index.store import AddOutcome, Store, sha256_text
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
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

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

        degraded = True
        if self.settings.llm_enabled:
            degraded = not self._enrich(user_id, session_id, plan, t0)
        return outcome, degraded

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

    def _enrich(
        self, user_id: str, session_id: str, plan: AddPlan, t0: float
    ) -> bool:
        """L3 enrichment hook (experience cards / episode summary).

        P1 ships without LLM enrichment. When enabled, this pass must:
          * select only chunks likely to carry reusable experience;
          * cache by content hash so retries never pay twice;
          * respect the internal deadline and return False on any failure.

        It never raises, so Add cannot be failed by enrichment.
        """
        deadline = t0 + self.settings.add_deadline_seconds
        try:
            if time.monotonic() >= deadline:
                log.warning(
                    "enrichment skipped: deadline reached",
                    extra={"ctx": {"user_id": user_id, "session_id": session_id}},
                )
                return False
            # Placeholder for P3: card extraction runs here.
            return True
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "enrichment failed; raw memory retained",
                extra={"ctx": {"user_id": user_id, "error": str(exc)}},
            )
            return False
