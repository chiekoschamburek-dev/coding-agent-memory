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
            for memory_id, score in self.store.bm25_search(user_id, probe, per_channel):
                if score > lexical.get(memory_id, float("-inf")):
                    lexical[memory_id] = score
        merge("lexical", sorted(lexical.items(), key=lambda kv: -kv[1])[:per_channel])

        # Channel 2: exact identifier match.
        entity_ranked = self.store.entity_search(user_id, plan.entities, per_channel)
        merge("entity", entity_ranked)

        # Channel 3: dense similarity. Catches paraphrase that shares no
        # literal identifier with the question. The query is embedded here, but
        # memory vectors were embedded during Add, so nothing is generated at
        # search time.
        dense_ranked = self._dense_recall(user_id, plan, per_channel)
        if dense_ranked:
            merge("dense", dense_ranked)

        # Channel 4: recency. Cheap, query-independent prior; only applied when
        # we actually have timestamps to compare.
        newest = self.store.max_ts(user_id)
        if newest > 0:
            recency = self._recency_ranking(user_id, newest, per_channel)
            merge("recency", recency)

        ordered = sorted(
            candidates.values(), key=lambda c: (-c.rrf, c.memory_id)
        )[: self.settings.candidate_pool]

        log.info(
            "recall",
            extra={
                "ctx": {
                    "user_id": user_id,
                    "intent": plan.intent,
                    "probes": len(plan.probes),
                    "entities": sum(len(v) for v in plan.entities.values()),
                    "candidates": len(ordered),
                    "channels": {
                        name: sum(1 for c in candidates.values() if name in c.channels)
                        for name in CHANNEL_WEIGHTS
                    },
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                }
            },
        )
        return ordered

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
