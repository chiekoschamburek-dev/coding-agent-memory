"""Evidence scoring and assembly.

Two jobs, both constrained by the design invariants.

**Scoring.** The platform takes a token-counted prefix of our payload in the
order we return, so ranking is effectively the score. We combine RRF (which
channels agreed), exact identifier evidence (per-type weights), structural
kind/intent alignment, and a superseded penalty. Recency only ever nudges — it
never filters, because trajectories do not tell us the repository's current
state.

**Assembly.** Sessions are ranked and chunks are the evidence: one session is one
piece of prior work, so its best chunk *is* its score, and the noise gate is
applied to that rather than to whichever chunk happens to come next. Within the
session we consider most likely to hold the answer, the chunk recording what the
session *did* leads, because a trajectory's reads outscore its single edit on
every channel and would otherwise take the only slots.

The content of every returned item is a verbatim span of what Add already stored.
Search performs no generation: truncation to fit the token budget is selection and
is allowed, rewriting is not. Items are shaped as a pyramid — a full form for the
head of the list, a pointer form for the tail — so the budget buys coverage
without truncating the strongest evidence.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Sequence

from ..core.config import Settings
from ..core.tokens import count_tokens, truncate_to_tokens
from ..index.store import MemoryRow, Store
from .query import QueryPlan
from .retriever import (
    COVERAGE_WEIGHT_IN_FINAL,
    ENTITY_WEIGHT_IN_FINAL,
    RRF_WEIGHT_IN_FINAL,
    STRENGTH_WEIGHT_IN_FINAL,
    Candidate,
)

# Which structural kinds a question most likely needs, keyed by detected intent.
# Adding "code" to the debug table was tried and measured worse on what the
# answer model actually reads (recall@10 0.7214 -> 0.7115, mean prefix
# `decidable` 0.408 -> 0.383; only MRR rose). Code is 81% of this corpus's
# chunks, so a kind-level bonus cannot isolate the tool-call records it was
# meant to favour -- that discriminator has to be per line.
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


# Lines that record what a session DID, rather than what it discussed. These are
# the highest-value evidence in an engineering trajectory, and they are written as
# tool-call JSON or patch syntax, so they match the issue wording poorly and are
# skipped by term-density selection alone.
# Unambiguous markers only. A generic "^[+-]" rule was tried and dropped: diff
# bodies indent after the sign ("+        code"), so it missed real diff lines
# while matching markdown bullets ("- item"). Locating a diff by its headers is
# enough, because a window is a contiguous span and the changed lines come with
# the header that pulled the window there.
_OPERATIVE_RES = (
    re.compile(r"\[tool (?:Edit|Write|MultiEdit|NotebookEdit)\]"),
    re.compile(r"^diff --git "),
    re.compile(r"^\+\+\+ b/"),
    re.compile(r"^--- a/"),
    re.compile(r"^@@ "),
    re.compile(r"has been updated"),
)


def _operative_score(line: str) -> int:
    return sum(1 for pattern in _OPERATIVE_RES if pattern.search(line))


def _has_operative_line(text: str) -> bool:
    return any(
        pattern.search(line) for line in text.splitlines() for pattern in _OPERATIVE_RES
    )


def _role_ordered(
    group: list[Candidate], sources: dict[int, str]
) -> list[Candidate]:
    """Move the best operative candidate to the front of one session's slice.

    A session yields tens of tool-call chunks that are near-identical to each
    other — the same file read five times and edited once — and they score
    alike on every channel because the paths carry the term weight and the verb
    carries none. Ordered by score alone, the edit can sit eighth in its own
    session and never reach a slot.
    """
    index = next(
        (
            i
            for i, cand in enumerate(group)
            if _has_operative_line(sources[cand.memory_id])
        ),
        None,
    )
    if index in (None, 0):
        return group
    return [group[index], *group[:index], *group[index + 1 :]]


def _select_span(
    text: str,
    budget_tokens: int,
    query_terms: Sequence[str],
    operative_weight: float = 0.0,
) -> tuple[str, bool]:
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
            if operative_weight:
                # What the session changed is worth more than what it said: an
                # edit or a diff line is the answer to "how was this handled",
                # while surrounding prose often restates the issue.
                score += operative_weight * _operative_score(lines[end])
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
    """Build the ranked ``data`` payload, respecting gates and token budget.

    Sessions are the ranking unit and chunks the evidence unit. ``scored`` is
    already in descending score order, so grouping it by session preserves both
    the session order (each session's head is its best chunk) and the order
    within a session. The noise gate then applies to a session's *best* evidence
    rather than to whichever of its chunks happens to come next.
    """
    if not scored or top_k <= 0:
        return []

    chunk_ids = [
        memories[c.memory_id].chunk_id
        for c in scored
        if c.memory_id in memories and memories[c.memory_id].chunk_id is not None
    ]
    chunk_map = store.fetch_chunks(user_id, chunk_ids)

    groups: dict[str, list[Candidate]] = {}
    order: list[str] = []
    for cand in scored:
        memory = memories.get(cand.memory_id)
        if memory is None:
            continue
        if memory.session_id not in groups:
            groups[memory.session_id] = []
            order.append(memory.session_id)
        groups[memory.session_id].append(cand)

    source_of = {
        cand.memory_id: (
            chunk_map[memories[cand.memory_id].chunk_id].text
            if memories[cand.memory_id].chunk_id in chunk_map
            else memories[cand.memory_id].text
        )
        for cand in scored
        if cand.memory_id in memories
    }

    budget = settings.evidence_budget_tokens
    used = 0
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    sessions_used = 0
    session_cap = max(1, settings.max_evidence_per_session)
    max_sessions = settings.evidence_max_sessions  # 0 = unlimited
    # Emission order is the relevance order we assert, so scores are clamped to
    # it: a second chunk of the decisive session can score below the head of a
    # weaker session while still being the better thing to show first.
    ceiling = float("inf")

    for session_index, session_id in enumerate(order):
        if max_sessions and sessions_used >= max_sessions:
            break
        if len(items) >= top_k:
            break

        group = groups[session_id]
        # Noise gate: stop rather than pad the answer model's prefix with
        # same-repository distractors.
        if group[0].final < settings.min_evidence_score and len(
            items
        ) >= settings.min_evidence_count:
            break

        promote = settings.evidence_operative_promotion
        if promote < 0 or session_index < promote:
            group = _role_ordered(group, source_of)

        taken = 0
        for cand in group:
            if taken >= session_cap or len(items) >= top_k:
                break

            memory = memories[cand.memory_id]
            body_source = source_of[cand.memory_id]
            full_form = len(items) < settings.evidence_full_count
            budget_for_item = (
                settings.evidence_item_tokens if full_form else settings.evidence_ptr_tokens
            )

            # content is a verbatim span of the stored memory. No header, no labels,
            # no rewriting: the contract states returned content is preserved
            # verbatim for audit, and the schema's example is plain remembered text.
            # Identifiers live in the index as retrieval keys, not in the payload,
            # and the source timestamp belongs in `created_at`, not in the text.
            content, truncated = _select_span(
                body_source,
                budget_for_item,
                plan.keywords,
                settings.evidence_operative_weight,
            )
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

            score = min(cand.final, ceiling)
            ceiling = score
            items.append(
                EvidenceItem(
                    memory_id=memory.id,
                    content=content,
                    score=round(score, 6),
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
            taken += 1

        if taken:
            sessions_used += 1

    # The contract requires a higher score to mean stronger relevance, and
    # emission order is the relevance order we assert, so decay by position by
    # more than the six-decimal rounding can absorb: a 1e-7 relative nudge rounds
    # back to the same value as its neighbour and yields two equal scores.
    for idx, item in enumerate(items):
        item.score = round(max(item.score - idx * 1e-5, 1e-6), 6)
    return items
