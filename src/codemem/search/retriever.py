"""Multi-channel recall and rank fusion.

Three channels run per query, each with a different failure mode, which is the
point: under same-repository noise all candidates share vocabulary and style, so
any single similarity signal saturates.

* ``lexical``  — FTS5 BM25 over text + identifier sub-tokens. Strong on exact
  error strings, paths and identifiers.
* ``entity``   — exact identifier equality with per-type weights. Immune to
  stylistic similarity; this is the channel that survives high noise.
* ``recency``  — weak prior favouring recent sessions, applied as a weight
  rather than a filter (we are never told the repository's current state).

Channels are fused with Reciprocal Rank Fusion, which needs no score
calibration across channels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.config import Settings
from ..core.logging import get_logger
from ..index.store import MemoryRow, Store
from .query import QueryPlan

log = get_logger("codemem.search.retriever")

# Channel weights for the *rank* fusion. Recency is deliberately near-tie-breaker
# weight: it carries no semantic signal, so it must not be able to outvote how
# strongly a memory actually matches. Leaving it at peer weight lets a merely
# recent memory displace a decisive lexical match, because rank fusion compresses
# large score gaps into small rank differences.
#
# ``dense`` sits below lexical and entity on purpose. In the Coding track every
# candidate comes from the same repository, so embedding similarity is uniformly
# high and comparatively uninformative; the dense channel exists to catch
# paraphrase that shares no identifiers, not to lead the ranking.
CHANNEL_WEIGHTS = {
    "lexical": 1.0,
    "entity": 1.15,  # exact identifiers are the most trustworthy signal
    "dense": 0.70,
    "recency": 0.08,
}

# Relative contribution of the fused rank vs. the raw match strength. Rank fusion
# discards magnitude, so a memory that wins BM25 by 5x looks almost identical to
# one that barely matches. The strength term restores that information.
RRF_WEIGHT_IN_FINAL = 0.40
COVERAGE_WEIGHT_IN_FINAL = 0.15
STRENGTH_WEIGHT_IN_FINAL = 0.30
ENTITY_WEIGHT_IN_FINAL = 0.15


@dataclass(slots=True)
class Candidate:
    memory_id: int
    channels: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    rrf: float = 0.0
    final: float = 0.0

    def add(self, channel: str, rank: int, score: float, weight: float, k: int) -> None:
        self.channels[channel] = rank
        self.channel_scores[channel] = score
        self.rrf += weight * (1.0 / (k + rank))


class Retriever:
    def __init__(self, settings: Settings, store: Store, embedder=None) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder

    def recall(self, user_id: str, plan: QueryPlan) -> list[Candidate]:
        t0 = time.monotonic()
        per_channel = self.settings.recall_per_channel
        # Session-major mode pulls a deeper slice per channel, so the
        # per-session cap below has more than a session or two to choose from.
        # The shallow slice is the legacy entry-major behaviour.
        depth = self.settings.recall_channel_depth or per_channel
        k = self.settings.rrf_k
        candidates: dict[int, Candidate] = {}

        def merge(channel: str, ranked: list[tuple[int, float]]) -> None:
            weight = CHANNEL_WEIGHTS.get(channel, 0.5)
            for rank, (memory_id, score) in enumerate(ranked, start=1):
                if memory_id <= 0:
                    continue
                cand = candidates.get(memory_id)
                if cand is None:
                    cand = Candidate(memory_id=memory_id)
                    candidates[memory_id] = cand
                if channel in cand.channels:
                    continue  # keep the best rank per channel
                cand.add(channel, rank, score, weight, k)

        # Channel 1: lexical BM25. Query each probe and fuse by best rank.
        lexical: dict[int, float] = {}
        for probe in plan.probes:
            for memory_id, score in self.store.bm25_search(user_id, probe, depth):
                if score > lexical.get(memory_id, float("-inf")):
                    lexical[memory_id] = score
        merge("lexical", sorted(lexical.items(), key=lambda kv: -kv[1])[:depth])

        # Channel 2: exact identifier match.
        entity_ranked = self.store.entity_search(user_id, plan.entities, depth)
        merge("entity", entity_ranked)

        # Channel 3: dense similarity. Catches paraphrase that shares no
        # literal identifier with the question. The query is embedded here, but
        # memory vectors were embedded during Add, so nothing is generated at
        # search time.
        dense_ranked = self._dense_recall(user_id, plan, depth)
        if dense_ranked:
            merge("dense", dense_ranked)

        # Channel 4: recency. Cheap, query-independent prior; only applied when
        # we actually have timestamps to compare.
        newest = self.store.max_ts(user_id)
        if newest > 0:
            recency = self._recency_ranking(user_id, newest, depth)
            merge("recency", recency)

        ordered = sorted(candidates.values(), key=lambda c: (-c.rrf, c.memory_id))
        per_session = self.settings.candidate_per_session
        pool_sessions: int | None = None
        if per_session > 0:
            ordered, pool_sessions = self._cap_per_session(user_id, ordered, per_session)
        ordered = ordered[: self.settings.candidate_pool]

        log.info(
            "recall",
            extra={
                "ctx": {
                    "user_id": user_id,
                    "intent": plan.intent,
                    "probes": len(plan.probes),
                    "entities": sum(len(v) for v in plan.entities.values()),
                    "candidates": len(ordered),
                    "pool_sessions": pool_sessions,
                    "channels": {
                        name: sum(1 for c in candidates.values() if name in c.channels)
                        for name in CHANNEL_WEIGHTS
                    },
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                }
            },
        )
        return ordered

    def _cap_per_session(
        self, user_id: str, ordered: list[Candidate], per_session: int
    ) -> tuple[list[Candidate], int]:
        """Keep at most ``per_session`` entries per session, order preserved.

        A session holds ~96 entries on the proxy corpus, so an entry-counted
        pool is dominated by whichever few sessions matched first: pool 300
        spanned ~3 sessions and rerank_top_n 120 barely 1.3. Capping per session
        spreads the same quota over far more sessions, which is what the
        session-level ranking downstream needs to have anything to order.
        """
        sessions = self.store.session_map(user_id, [c.memory_id for c in ordered])
        kept: list[Candidate] = []
        counts: dict[str, int] = {}
        for cand in ordered:
            session = sessions.get(cand.memory_id)
            if session is None:  # unknown session: never drop evidence for it
                kept.append(cand)
                continue
            seen = counts.get(session, 0)
            if seen >= per_session:
                continue
            counts[session] = seen + 1
            kept.append(cand)
        return kept, len(counts)

    def _dense_recall(
        self, user_id: str, plan: QueryPlan, limit: int
    ) -> list[tuple[int, float]]:
        """Nearest memories by embedding similarity, within one user.

        Returns empty when the dense channel is disabled or the encoder is
        unavailable, so the system degrades to its lexical channels rather than
        failing. A similarity floor avoids injecting low-confidence neighbours,
        which in a same-repository corpus are plentiful and uninformative.
        """
        instance = self.embedder
        if instance is None or not instance.available:
            return []
        # The question plus each probe: the raw question alone can be short and
        # vague, while option-derived probes widen the topic space.
        targets = [plan.query] + [p for p in plan.probes if p != plan.query][:3]
        vectors = instance.embed(targets)
        if not vectors:
            return []

        best: dict[int, float] = {}
        for vector in vectors:
            for memory_id, score in self.store.dense_search(user_id, vector, limit):
                if score < self.settings.dense_min_similarity:
                    continue
                if score > best.get(memory_id, float("-inf")):
                    best[memory_id] = score
        return sorted(best.items(), key=lambda kv: -kv[1])[:limit]

    def _recency_ranking(
        self, user_id: str, newest: int, limit: int
    ) -> list[tuple[int, float]]:
        """Rank a slice of the most recent memories, decaying by age."""
        rows = self._recent(user_id, limit)
        half_life_ms = 1000.0 * 60.0 * 60.0 * 24.0 * 30.0  # 30 days
        ranked: list[tuple[int, float]] = []
        for memory_id, ts in rows:
            age = max(0, newest - (ts or 0))
            decay = 0.5 ** (age / half_life_ms)
            ranked.append((memory_id, decay))
        ranked.sort(key=lambda kv: -kv[1])
        return ranked

    def _recent(self, user_id: str, limit: int) -> list[tuple[int, int]]:
        with self.store._read() as conn:  # noqa: SLF001 - same package boundary
            rows = conn.execute(
                "SELECT id, coalesce(ts, 0) AS ts FROM memory WHERE user_id = ?"
                " ORDER BY coalesce(ts, 0) DESC, id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [(int(r["id"]), int(r["ts"])) for r in rows]

    def load(self, user_id: str, candidates: list[Candidate]) -> dict[int, MemoryRow]:
        return self.store.fetch_memories(user_id, [c.memory_id for c in candidates])
