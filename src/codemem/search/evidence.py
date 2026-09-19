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
from typing import Sequence

from ..add.entities import ENTITY_WEIGHT
from ..core.config import Settings
from ..core.tokens import count_tokens, truncate_to_tokens
from ..index.store import ChunkRow, MemoryRow, Store
from .query import QueryPlan
from .retriever import (
    COVERAGE_WEIGHT_IN_FINAL,
    ENTITY_WEIGHT_IN_FINAL,
    RRF_WEIGHT_IN_FINAL,
    STRENGTH_WEIGHT_IN_FINAL,
    Candidate,
)

# Which structural kinds a question most likely needs, keyed by detected intent.
INTENT_KIND_BONUS: dict[str, dict[str, float]] = {
    "debug": {"stacktrace": 1.18, "diff": 1.15, "test": 1.10, "log": 1.08, "cmd": 1.04},
    "develop": {"diff": 1.18, "code": 1.12, "prose": 1.06, "test": 1.05},
    "general": {},
}

@dataclass(slots=True)
class EvidenceItem:
    memory_id: int
    content: str
    score: float
    created_at: str | None
    tokens: int
    truncated: bool = False
    superseded: bool = False


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

    The blend includes a *strength* term on purpose. Pure rank fusion discards
    magnitude, so a memory that wins BM25 by five times looks almost identical
    to one that barely cleared the match — and a merely-recent, unrelated memory
    can then displace a decisive match. Restoring magnitude fixes that.
    """
    eligible = [
        c for c in candidates if any(name in c.channels for name in INFORMATIVE_CHANNELS)
    ]
    if not eligible:
        return []

    max_rrf = max((c.rrf for c in eligible), default=0.0) or 1.0
    max_entity = max(entity_match_by_memory.values(), default=0.0)

    # Per-channel maxima so each channel's magnitude is normalized within this
    # query, which makes channels comparable without cross-query calibration.
    channel_max: dict[str, float] = {}
    for cand in eligible:
        for name in INFORMATIVE_CHANNELS:
            score = cand.channel_scores.get(name)
            if score is not None:
                channel_max[name] = max(channel_max.get(name, 0.0), score)

    for cand in eligible:
        memory = memories.get(cand.memory_id)
        base = cand.rrf / max_rrf

        # Agreement across independent evidence channels. Recency is excluded:
        # a stale memory is not thereby relevant, nor a fresh one useful.
        coverage = sum(
            1 for name in INFORMATIVE_CHANNELS if name in cand.channels
        ) / len(INFORMATIVE_CHANNELS)

        # Best normalized match strength across the informative channels.
        strength = 0.0
        for name in INFORMATIVE_CHANNELS:
            score = cand.channel_scores.get(name)
            top = channel_max.get(name, 0.0)
            if score is not None and top > 0:
                strength = max(strength, score / top)

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

        combined = (
            RRF_WEIGHT_IN_FINAL * base
            + COVERAGE_WEIGHT_IN_FINAL * coverage
            + STRENGTH_WEIGHT_IN_FINAL * strength
            + ENTITY_WEIGHT_IN_FINAL * entity_signal
        )
        cand.final = combined * bonus * penalty

    scored = sorted(
        [c for c in eligible if c.final > 0], key=lambda c: (-c.final, c.memory_id)
    )
    top = scored[0].final if scored else 0.0
    if top > 0:
        for cand in scored:
            cand.final = cand.final / top
    return scored


def _iso_from_ms(value: int | None) -> str | None:
    """Format a source timestamp as ISO-8601, or None when the source had none.

    Only ever applied to a timestamp the platform supplied; we never substitute
    our own processing time for a missing source time.
    """
    if value is None:
        return None
    try:
        from datetime import datetime, timezone

        # Accept both seconds and milliseconds; the contract sends milliseconds.
        seconds = value / 1000.0 if abs(value) > 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OverflowError, OSError, ValueError):
        return None


def _select_span(text: str, budget_tokens: int, query_terms: Sequence[str]) -> tuple[str, bool]:
    """Choose a verbatim, line-aligned span of ``text`` within the token budget.

    Returns ``(span, truncated)``.

    Selection is by query-term density rather than by taking the opening lines.
    A real session trajectory is long, and its first few hundred tokens are often
    framing chatter while the diagnosis or the fix sits in the middle. Choosing
    the densest window keeps the most useful *verbatim* text, which matters
    because we may not rephrase the memory to make it fit.

    Ties resolve to the earliest window, so a memory with no term overlap still
    yields its opening rather than an arbitrary slice.
    """
    if budget_tokens <= 0:
        return "", True
    if count_tokens(text) <= budget_tokens:
        return text, False

    lines = text.splitlines()
    if not lines:
        span = truncate_to_tokens(text, budget_tokens)
        return span, True

    lowered = [line.lower() for line in lines]
    costs = [count_tokens(line) + 1 for line in lines]
    terms = [t for t in query_terms if t]

    best_start = best_end = 0
    best_score = -1.0
    for start in range(len(lines)):
        cost = 0
        score = 0.0
        end = start
        while end < len(lines):
            cost += costs[end]
            if cost > budget_tokens:
                break
            score += sum(1 for term in terms if term in lowered[end])
            end += 1
        # Strict '>' keeps the earliest window on a tie.
        if end > start and score > best_score:
            best_score, best_start, best_end = score, start, end

    if best_end <= best_start:
        return truncate_to_tokens(text, budget_tokens), True

    span = "\n".join(lines[best_start:best_end]).strip()
    if not span:
        return truncate_to_tokens(text, budget_tokens), True
    # Mark elision explicitly rather than pretending the span is the whole
    # memory. The marks are conventional truncation indicators, not content.
    if best_start > 0:
        span = "…\n" + span
    if best_end < len(lines):
        span = span + "\n…"
    return span, True


def assemble(
    settings: Settings,
    store: Store,
    user_id: str,
    plan: QueryPlan,
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
    per_session: dict[str, int] = {}
    session_cap = max(1, settings.max_evidence_per_session)

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

        # Diversity cap. Several chunks of one session are one piece of prior
        # work; letting one dominate the list starves other sessions, which is
        # what recall@k actually measures. Skipped items still leave their slot
        # available for the next session rather than shortening the list.
        if per_session.get(memory.session_id, 0) >= session_cap:
            continue

        chunk: ChunkRow | None = (
            chunk_map.get(memory.chunk_id) if memory.chunk_id is not None else None
        )

        body_source = chunk.text if chunk is not None else memory.text
        full_form = rank < settings.evidence_full_count
        budget_for_item = (
            settings.evidence_item_tokens if full_form else settings.evidence_ptr_tokens
        )

        # content is a verbatim span of the stored memory. No header, no labels,
        # no rewriting: the contract states returned content is preserved
        # verbatim for audit, and the schema's example is plain remembered text.
        # Identifiers live in the index as retrieval keys, not in the payload,
        # and the source timestamp belongs in `created_at`, not in the text.
        content, truncated = _select_span(body_source, budget_for_item, plan.keywords)
        stamped = _iso_from_ms(memory.ts)

        # Repeated text across sessions adds no new evidence but would consume
        # the answer model's context.
        if content in seen:
            continue
        seen.add(content)

        item_tokens = count_tokens(content)
        if items and used + item_tokens > budget:
            break
        used += item_tokens
        per_session[memory.session_id] = per_session.get(memory.session_id, 0) + 1

        items.append(
            EvidenceItem(
                memory_id=memory.id,
                content=content,
                score=round(cand.final, 6),
                # The contract defines created_at as the memory's source time or
                # persistence time. We prefer the source timestamp supplied by
                # the platform; only when the source had none do we fall back to
                # our own write time, and the two are never conflated.
                created_at=stamped or memory.created_at,
                tokens=item_tokens,
                truncated=truncated,
                superseded=memory.superseded_by is not None,
            )
        )

    # Guarantee a strictly decreasing score sequence so the returned order and
    # the returned scores can never disagree, whatever the tie situation.
    for idx, item in enumerate(items):
        item.score = round(max(item.score, 1e-6) * (1.0 - idx * 1e-7), 6)
    return items
