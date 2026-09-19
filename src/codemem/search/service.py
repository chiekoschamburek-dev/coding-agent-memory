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
from ..index.store import MemoryRow, Store
from .evidence import EvidenceItem, assemble, score_candidates
from .query import plan_query
from .retriever import Retriever

log = get_logger("codemem.search")


class SearchPipeline:
    def __init__(
        self, settings: Settings, store: Store, embedder=None, reranker=None
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = Retriever(settings, store, embedder=embedder)
        self.reranker = reranker

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

        # Rerank the head of the fused list by reading each (question, memory)
        # pair jointly. Only the top slice is reranked: a cross-encoder is
        # O(pool) forward passes and the tail could not reach the answer model
        # anyway.
        reranked = self._rerank(user_id, plan, scored, memories)

        items = assemble(
            self.settings,
            self.store,
            user_id,
            reranked,
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
                    "reranked": bool(self.reranker and reranked is not scored),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                }
            },
        )
        return items

    def _rerank(
        self,
        user_id: str,
        plan: QueryPlan,
        scored: list[Candidate],
        memories: dict[int, MemoryRow],
    ) -> list[Candidate]:
        """Blend cross-encoder scores into the fused ranking, in place.

        Returns ``scored`` unchanged when reranking is disabled or unavailable,
        so the pipeline degrades to the recall ordering.
        """
        if not self.reranker or not scored:
            return scored
        head = scored[: self.settings.rerank_top_n]
        texts: list[str] = []
        for cand in head:
            memory = memories.get(cand.memory_id)
            if memory is None:
                texts.append("")
                continue
            # Include the deterministic header fields so the cross-encoder sees
            # the identifiers too, not only prose.
            body = memory.text[: self.settings.rerank_max_chars]
            texts.append(f"{memory.structural_kind}: {body}")

        scores = self.reranker.score(plan.query, texts)
        if not scores or len(scores) != len(head):
            return scored

        from ..rerank import sigmoid

        # Temperature-scaled sigmoid, NOT max-normalised.
        #
        # Normalising by the maximum made every score depend on which items
        # happened to be in the reranked head, so changing rerank_top_n changed
        # the ranking non-monotonically (measured: top_n=30 scored MRR 0.789,
        # top_n=60 scored 0.772, top_n=120 scored 0.818 — an incoherent
        # sequence). A fixed temperature keeps the mapping absolute: the same
        # logit always yields the same contribution, whatever the pool size.
        temperature = self.settings.rerank_temperature
        normalized = [sigmoid(s / temperature) for s in scores]

        weight = self.settings.rerank_weight
        for cand, rerank_score in zip(head, normalized):
            cand.final = (1.0 - weight) * cand.final + weight * rerank_score

        ranked = sorted(scored, key=lambda c: (-c.final, c.memory_id))
        top_final = ranked[0].final if ranked else 0.0
        if top_final > 0:
            for cand in ranked:
                cand.final = cand.final / top_final
        return ranked
