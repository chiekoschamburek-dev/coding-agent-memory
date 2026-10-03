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
from ..core.tokens import count_tokens
from ..index.store import MemoryRow, Store
from .evidence import (
    INFORMATIVE_CHANNELS,
    EvidenceItem,
    _iso_from_ms,
    _select_span,
    assemble,
    score_candidates,
)
from .query import plan_query
from .retriever import Retriever

log = get_logger("codemem.search")


class SearchPipeline:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        embedder=None,
        reranker=None,
        listwise=None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = Retriever(settings, store, embedder=embedder)
        self.reranker = reranker
        self.listwise = listwise

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

        scored = score_candidates(
            candidates,
            memories,
            plan,
            entity_match_by_memory,
            dense_only_min_similarity=(
                self.settings.dense_eligible_min_similarity
                if self.settings.dense_eligible
                else None
            ),
            dense_only_max=self.settings.dense_eligible_max,
            operative_weight=self.settings.operative_rank_weight,
        )

        # Rerank the head of the fused list by reading each (question, memory)
        # pair jointly. Only the top slice is reranked: a cross-encoder is
        # O(pool) forward passes and the tail could not reach the answer model
        # anyway.
        reranked = self._rerank(user_id, plan, scored, memories)

        # Final stage: an LLM judges the head of the list comparatively. A
        # cross-encoder scores each pair in isolation, so it cannot tell that
        # forty candidates all come from the same repository and only one
        # explains the failure; a listwise read can. Scores only -- no model
        # text is ever returned (see the module docstring).
        reranked = self._listwise_rerank(plan, reranked, memories)

        items = assemble(
            self.settings,
            self.store,
            user_id,
            plan,
            reranked,
            memories,
            top_k=limit,
        )
        items = self._dense_fill(user_id, plan, candidates, memories, items)

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

    def _dense_fill(
        self,
        user_id: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        memories: dict[int, MemoryRow],
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        """Append a few memories that only the dense channel reached.

        The append-only form of dense-only admission, deliberately weaker than
        scoring them normally: nothing the lexical channels produced is revisited.
        Their order, their scores and the session slots they occupy all stay
        exactly as ``assemble`` decided.

        Three exclusions define what may be appended:

        * not already returned;
        * not from a session already represented — a second chunk of a session
          already shown costs tokens without adding a new lead;
        * not contested by ``lexical`` or ``entity``. That last one is what makes
        this a *fill* for the lexical channels' blind spot rather than a second
        chance for something they already ranked and left behind.

        It never fires on an empty result: the noise gate having abstained is a
        decision about the query, and appending to ``[]`` would undo it.

        Scores continue strictly decreasing from the tail, since emission order
        is the relevance order we assert.
        """
        settings = self.settings
        if not settings.dense_fill or not items or settings.dense_fill_max <= 0:
            return items

        floor = settings.dense_fill_min_similarity
        used_memory = {item.memory_id for item in items}
        used_session: set[str] = set()
        for item in items:
            memory = memories.get(item.memory_id)
            if memory is not None and memory.session_id:
                used_session.add(memory.session_id)

        fillable = [
            cand
            for cand in candidates
            if cand.memory_id not in used_memory
            and not any(name in cand.channels for name in INFORMATIVE_CHANNELS)
            and cand.channel_scores.get("dense", 0.0) >= floor
        ]
        fillable.sort(key=lambda c: (-c.channel_scores.get("dense", 0.0), c.memory_id))

        out = list(items)
        tail = out[-1].score if out else 0.0
        for cand in fillable:
            if len(out) - len(items) >= settings.dense_fill_max:
                break
            memory = memories.get(cand.memory_id)
            if memory is None or memory.session_id in used_session:
                continue
            content, truncated = _select_span(
                memory.text,
                settings.dense_fill_tokens,
                plan.keywords,
                settings.evidence_operative_weight,
            )
            if not content:
                continue
            # Stay below the last emitted score by more than the 6-decimal
            # rounding can absorb, or two neighbours collapse to one value.
            # Stop rather than emit a duplicate at the floor: `assemble` clamps
            # to 1e-6, so a long tail can leave no room left to decrease into.
            nxt = tail - 1e-5
            if nxt < 1e-6:
                break
            tail = nxt
            out.append(
                EvidenceItem(
                    memory_id=memory.id,
                    content=content,
                    score=round(tail, 6),
                    created_at=_iso_from_ms(memory.ts) or memory.created_at,
                    tokens=count_tokens(content),
                    truncated=truncated,
                    superseded=memory.superseded_by is not None,
                )
            )
            used_memory.add(memory.id)
            if memory.session_id:
                used_session.add(memory.session_id)
        return out

    def _rerank_body(self, memory: MemoryRow, plan: QueryPlan) -> str:
        """The text the cross-encoder reads for one memory.

        A character prefix is the wrong window on a long trajectory: it keeps
        whichever lines happen to come first, not the ones that match, and the
        listwise experiment already showed the judge gets better when it sees
        more of the relevant excerpt. With ``rerank_span_tokens`` set, the
        window is chosen the same way the returned item's content is — by
        query-term density with operative lines weighted — so the stage scores
        the part of the memory that is actually on topic.
        """
        budget = self.settings.rerank_span_tokens
        if budget <= 0:
            return memory.text[: self.settings.rerank_max_chars]
        span, _ = _select_span(
            memory.text,
            budget,
            plan.keywords,
            self.settings.evidence_operative_weight,
        )
        return span

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Map cross-encoder output onto the 0..1 scale the blend expects.

        Most cross-encoders emit an unbounded logit, which a fixed temperature
        turns into a contribution that does not depend on the pool. Some (the
        bge-reranker family) emit a 0..1 relevance score already; putting those
        through the same sigmoid flattens every candidate towards 0.5 and
        destroys the ordering, so they are taken as they come.
        """
        if self.settings.rerank_probability_scores:
            return [min(1.0, max(0.0, s)) for s in scores]
        from ..rerank import sigmoid

        temperature = self.settings.rerank_temperature
        return [sigmoid(s / temperature) for s in scores]

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
        if not scored:
            return scored
        if self.settings.rerank_session_level:
            return self._rerank_sessions(plan, scored, memories)
        if not self.reranker:
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
            texts.append(
                f"{memory.structural_kind}: {self._rerank_body(memory, plan)}"
            )

        scores = self.reranker.score(plan.query, texts)
        if not scores or len(scores) != len(head):
            return scored

        # Temperature-scaled sigmoid, NOT max-normalised.
        #
        # Normalising by the maximum made every score depend on which items
        # happened to be in the reranked head, so changing rerank_top_n changed
        # the ranking non-monotonically (measured: top_n=30 scored MRR 0.789,
        # top_n=60 scored 0.772, top_n=120 scored 0.818 — an incoherent
        # sequence). A fixed temperature keeps the mapping absolute: the same
        # logit always yields the same contribution, whatever the pool size.
        normalized = self._normalize_scores(scores)

        weight = self.settings.rerank_weight
        for cand, rerank_score in zip(head, normalized):
            cand.final = (1.0 - weight) * cand.final + weight * rerank_score

        ranked = sorted(scored, key=lambda c: (-c.final, c.memory_id))
        top_final = ranked[0].final if ranked else 0.0
        if top_final > 0:
            for cand in ranked:
                cand.final = cand.final / top_final
        return ranked

    def _rerank_sessions(
        self,
        plan: QueryPlan,
        scored: list[Candidate],
        memories: dict[int, MemoryRow],
    ) -> list[Candidate]:
        """Score one representative document per session, not per entry.

        The entry-level stage spends its whole budget inside whichever sessions
        happen to fill the pool — 1.3 of them on the proxy corpus — so it can
        only re-order chunks within a session. Judging one document per session
        spends the same number of forward passes on the question the payload
        actually asks: which session comes first.

        The score is applied to every candidate of that session, so the
        within-session order that ``assemble`` preserves is left intact and only
        the session order moves.
        """
        if not self.reranker:
            return scored

        head = scored[: self.settings.rerank_top_n]
        groups: dict[str, list[Candidate]] = {}
        order: list[str] = []
        loose: list[Candidate] = []
        for cand in head:
            memory = memories.get(cand.memory_id)
            session = memory.session_id if memory is not None else None
            if session is None:
                loose.append(cand)
                continue
            if session not in groups:
                groups[session] = []
                order.append(session)
            groups[session].append(cand)

        def document(cand: Candidate) -> str:
            memory = memories.get(cand.memory_id)
            if memory is None:
                return ""
            return f"{memory.structural_kind}: {self._rerank_body(memory, plan)}"

        # ``scored`` is sorted, so a session's first entry is its strongest.
        # That is the representative the cross-encoder reads.
        targets: list[list[Candidate]] = [[cand] for cand in loose]
        texts = [document(cand) for cand in loose]
        for session in order:
            group = groups[session]
            texts.append(document(group[0]))
            targets.append(group)

        scores = self.reranker.score(plan.query, texts)
        if not scores or len(scores) != len(targets):
            return scored

        normalized = self._normalize_scores(scores)

        weight = self.settings.rerank_weight
        for group, rerank_score in zip(targets, normalized):
            for cand in group:
                cand.final = (1.0 - weight) * cand.final + weight * rerank_score

        ranked = sorted(scored, key=lambda c: (-c.final, c.memory_id))
        top_final = ranked[0].final if ranked else 0.0
        if top_final > 0:
            for cand in ranked:
                cand.final = cand.final / top_final
        return ranked

    def _listwise_rerank(
        self,
        plan: QueryPlan,
        scored: list[Candidate],
        memories: dict[int, MemoryRow],
    ) -> list[Candidate]:
        """Blend LLM listwise relevance into the ranking, in place.

        Returns ``scored`` unchanged when listwise reranking is disabled or
        unavailable, so the pipeline falls back to the fused/cross-encoder order.

        The question text sent to the judge is the platform's query, and the
        documents are stored memory text. Nothing the model produces is returned:
        only the parsed scores are used, which keeps Search on the right side of
        the "return memory evidence, do not generate" rule.
        """
        if not self.listwise or not scored:
            return scored

        head = scored[: self.settings.listwise_max_candidates]
        documents: list[str] = []
        for cand in head:
            memory = memories.get(cand.memory_id)
            documents.append(memory.text if memory is not None else "")
        if not any(documents):
            return scored

        raw_scores = self.listwise.score(plan.query, documents)
        if not raw_scores:
            return scored

        from ..listwise import normalise

        judged = normalise(raw_scores)
        if len(judged) != len(head):
            return scored

        weight = self.settings.listwise_weight
        for cand, score in zip(head, judged):
            if score is None:
                continue  # unjudged: leave the existing score alone
            cand.final = (1.0 - weight) * cand.final + weight * score

        ranked = sorted(scored, key=lambda c: (-c.final, c.memory_id))
        top = ranked[0].final if ranked else 0.0
        if top > 0:
            for cand in ranked:
                cand.final = cand.final / top
        return ranked
