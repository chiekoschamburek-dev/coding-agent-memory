"""Evidence scoring and assembly.

Two jobs, both constrained by the design invariants.

**Scoring.** The platform takes a token-counted prefix of our payload in the
order we return, so ranking is effectively the score. We combine RRF (which
channels agreed), exact identifier evidence (per-type weights), structural
kind/intent alignment, and a superseded penalty. Recency only ever nudges — it
never filters, because trajectories do not tell us the repository's current
state.

**Assembly.** The content of every returned item is built *only* from what Add
already stored: verbatim excerpts plus deterministically extracted identifiers.
Search performs no generation. The fixed header template is formatting, not new
claims — every field is literally a value computed during Add.

Items are shaped as a pyramid: a full form for the head of the list and a
pointer form for the tail, so the token budget buys maximum coverage without
truncating the strongest evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..add.entities import ENTITY_WEIGHT
from ..core.config import Settings
from ..core.tokens import count_tokens, truncate_to_tokens
from ..index.store import ChunkRow, MemoryRow, Store
from .query import QueryPlan
from .retriever import Candidate

# Which structural kinds a question most likely needs, keyed by detected intent.
INTENT_KIND_BONUS: dict[str, dict[str, float]] = {
    "debug": {"stacktrace": 1.18, "diff": 1.15, "test": 1.10, "log": 1.08, "cmd": 1.04},
    "develop": {"diff": 1.18, "code": 1.12, "prose": 1.06, "test": 1.05},
    "general": {},
}

# Identifiers worth echoing in the header, in priority order.
_HEADER_ENTITIES = (
    "file_path",
    "file_name",
    "symbol",
    "exception",
    "test",
    "issue_id",
    "cmd",
    "pkg",
)


@dataclass(slots=True)
class EvidenceItem:
    memory_id: int
    content: str
    score: float
    created_at: str | None
    tokens: int


# Channels that constitute actual evidence. Recency is deliberately excluded:
# being recent says nothing about relevance, and counting it would let an
# irrelevant-but-new memory clear the noise gate on freshness alone.
INFORMATIVE_CHANNELS = ("lexical", "entity")


def _kind_bonus(kind: str, intent: str) -> float:
    return INTENT_KIND_BONUS.get(intent, {}).get(kind, 1.0)


def score_candidates(
    candidates: list[Candidate],
    memories: dict[int, MemoryRow],
    plan: QueryPlan,
    entity_match_by_memory: dict[int, float],
) -> list[Candidate]:
    """Assign a normalized, strictly decreasing ``final`` score to candidates.

    Two gates, both absolute rather than relative:

    1. **Evidence requirement.** Scores are normalized against the best
       candidate, so the top item would always be 1.0 — relative gating could
       never drop anything, and a query with no relevant memory would still
       return one irrelevant item. So a candidate is only eligible if at least
       one *informative* channel (lexical or entity) found it. Being recent is
       not evidence.
    2. **Noise threshold** (applied by the caller) trims the tail.

    ``entity_match_by_memory`` holds IDF-weighted identifier scores, normalized
    within this query so questions of differing identifier density stay
    comparable.
    """
    eligible = [
        c for c in candidates if any(name in c.channels for name in INFORMATIVE_CHANNELS)
    ]
    if not eligible:
        return []

    max_rrf = max((c.rrf for c in eligible), default=0.0) or 1.0
    max_entity = max(entity_match_by_memory.values(), default=0.0)

    for cand in eligible:
        memory = memories.get(cand.memory_id)
        base = cand.rrf / max_rrf

        # Agreement across independent evidence channels. Recency is excluded:
        # a stale memory is not thereby relevant, nor a fresh one useful.
        coverage = sum(
            1 for name in INFORMATIVE_CHANNELS if name in cand.channels
        ) / len(INFORMATIVE_CHANNELS)

        # Normalized identifier evidence, damped: one strong match helps, but a
        # memory must not win on identifiers alone, since same-repository
        # distractors share paths too.
        entity_ratio = (
            entity_match_by_memory.get(cand.memory_id, 0.0) / max_entity
            if max_entity > 0
            else 0.0
        )
        entity_signal = entity_ratio ** 1.5

        kind = memory.structural_kind if memory is not None else "chunk"
        bonus = _kind_bonus(kind, plan.intent)

        # Superseded memories stay visible but rank lower: an old fix that a
        # later session replaced is still potentially the useful precedent.
        penalty = 0.65 if (memory is not None and memory.superseded_by) else 1.0

        cand.final = (0.45 * base + 0.20 * coverage + 0.35 * entity_signal) * bonus * penalty

    scored = sorted(
        [c for c in eligible if c.final > 0], key=lambda c: (-c.final, c.memory_id)
    )
    top = scored[0].final if scored else 0.0
    if top > 0:
        for cand in scored:
            cand.final = cand.final / top
    return scored


def _header(
    *,
    kind: str,
    lang: str | None,
    entities: list[tuple[str, str]],
    created_at: str | None,
    superseded: bool,
) -> str:
    label = kind or "chunk"
    if lang:
        label = f"{label} · {lang}"
    if superseded:
        label += " · superseded"
    lines = [f"[memory] {label}"]

    by_type: dict[str, list[str]] = {}
    for etype, value in entities:
        by_type.setdefault(etype, []).append(value)
    for etype in _HEADER_ENTITIES:
        values = by_type.get(etype)
        if not values:
            continue
        lines.append(f"[{etype}] " + ", ".join(values[:6]))
    if created_at:
        lines.append(f"[time] {created_at}")
    return "\n".join(lines)


def assemble(
    settings: Settings,
    store: Store,
    user_id: str,
    scored: list[Candidate],
    memories: dict[int, MemoryRow],
    *,
    top_k: int,
) -> list[EvidenceItem]:
    """Build the ranked ``data`` payload, respecting gates and token budget."""
    if not scored or top_k <= 0:
        return []

    chunk_ids = [
        memories[c.memory_id].chunk_id
        for c in scored
        if c.memory_id in memories and memories[c.memory_id].chunk_id is not None
    ]
    entity_map = store.entities_for_chunks(user_id, chunk_ids)
    chunk_map = store.fetch_chunks(user_id, chunk_ids)

    budget = settings.evidence_budget_tokens
    used = 0
    items: list[EvidenceItem] = []
    seen: set[str] = set()

    for rank, cand in enumerate(scored):
        if len(items) >= top_k:
            break

        memory = memories.get(cand.memory_id)
        if memory is None:
            continue

        # Noise gate: stop rather than pad the answer model's prefix with
        # same-repository distractors.
        if cand.final < settings.min_evidence_score and len(items) >= settings.min_evidence_count:
            break

        chunk: ChunkRow | None = (
            chunk_map.get(memory.chunk_id) if memory.chunk_id is not None else None
        )
        entities = entity_map.get(memory.chunk_id, []) if memory.chunk_id is not None else []

        head = _header(
            kind=memory.structural_kind,
            lang=memory.structural_lang,
            entities=entities,
            created_at=memory.created_at,
            superseded=memory.superseded_by is not None,
        )

        full_form = rank < settings.evidence_full_count
        body_budget = (
            settings.evidence_item_tokens if full_form else settings.evidence_ptr_tokens
        )
        body_room = max(24, body_budget - count_tokens(head))
        body_source = chunk.text if chunk is not None else memory.text
        body = truncate_to_tokens(body_source, body_room)
        content = f"{head}\n---\n{body}".strip()

        # Repeated text across sessions adds no new evidence but would consume
        # the answer model's context.
        if content in seen:
            continue
        seen.add(content)

        item_tokens = count_tokens(content)
        if items and used + item_tokens > budget:
            break
        used += item_tokens

        items.append(
            EvidenceItem(
                memory_id=memory.id,
                content=content,
                score=round(cand.final, 6),
                created_at=memory.created_at,
                tokens=item_tokens,
            )
        )

    # Guarantee a strictly decreasing score sequence so the returned order and
    # the returned scores can never disagree, whatever the tie situation.
    for idx, item in enumerate(items):
        item.score = round(max(item.score, 1e-6) * (1.0 - idx * 1e-7), 6)
    return items
