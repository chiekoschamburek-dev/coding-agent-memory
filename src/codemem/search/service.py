"""Search orchestration.

Reads only ``user_id``-scoped memory and never generates content: this pipeline
plans probes, recalls from several channels, fuses, scores, gates, and formats.
The LLM (when enabled, from P3) is permitted only to *score* existing memories;
it never writes text that is returned to the platform.
"""

from __future__ import annotations

import time

from ..core.config import Settings
from ..core.logging import get_logger
from ..index.store import Store
from .evidence import EvidenceItem, assemble, score_candidates
from .query import plan_query
from .retriever import Retriever

log = get_logger("codemem.search")


class SearchPipeline:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.retriever = Retriever(settings, store)

    def handle(
        self, *, user_id: str, query: str, options: list[str] | None, top_k: int
    ) -> list[EvidenceItem]:
        t0 = time.monotonic()
        limit = max(1, min(top_k, self.settings.max_top_k))

        plan = plan_query(query, options)
        candidates = self.retriever.recall(user_id, plan)
        if not candidates:
            log.info(
                "search empty",
                extra={"ctx": {"user_id": user_id, "reason": "no_candidates"}},
            )
            return []

        memories = self.retriever.load(user_id, candidates)

        # Exact-identifier evidence is computed per chunk by the store (IDF
        # weighted), then projected onto the memories that cover those chunks.
        chunk_scores = self.store.entity_match_scores(user_id, plan.entities)
        entity_match_by_memory: dict[int, float] = {}
        if chunk_scores:
            for cand in candidates:
                memory = memories.get(cand.memory_id)
                if memory is None or memory.chunk_id is None:
                    continue
                score = chunk_scores.get(memory.chunk_id, 0.0)
                if score > 0:
                    entity_match_by_memory[memory.id] = score

        scored = score_candidates(candidates, memories, plan, entity_match_by_memory)
        items = assemble(
            self.settings,
            self.store,
            user_id,
            scored,
            memories,
            top_k=limit,
        )

        log.info(
            "search done",
            extra={
                "ctx": {
                    "user_id": user_id,
                    "intent": plan.intent,
                    "top_k": limit,
                    "returned": len(items),
                    "tokens": sum(i.tokens for i in items),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                }
            },
        )
        return items
