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


def _operative_term(text: str, cap: int = 3) -> float:
    """0..1 grade of "this chunk records an action", from its own lines.

    Bounded rather than proportional: a chunk with one ``[tool Edit]`` line and a
    diff hunk with ten marker lines carry the same kind of evidence, and letting
    the count scale linearly would turn large diffs into outliers in the blend.
    """
    total = sum(_operative_score(line) for line in text.splitlines())
    if not total:
        return 0.0
    return min(1.0, total / cap)


def score_candidates(
    candidates: list[Candidate],
    memories: dict[int, MemoryRow],
    plan: QueryPlan,
    entity_match_by_memory: dict[int, float],
    *,
    dense_only_min_similarity: float | None = None,
    dense_only_max: int = 0,
    operative_weight: float = 0.0,
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

    ``operative_weight`` (0 = off) adds a fifth term, "this chunk records an
    action", and rescales the other four so the total stays on the same 0..1
    scale the noise gate is calibrated against.

    The blend includes a *strength* term on purpose. Pure rank fusion discards
    magnitude, so a memory that wins BM25 by five times looks almost identical
    to one that barely cleared the match — and a merely-recent, unrelated memory
    can then displace a decisive match. Restoring magnitude fixes that.

    Dense-only admission
    --------------------
    ``dense_only_min_similarity`` (``None`` keeps the behaviour above) admits a
    candidate that *only* the dense channel found. Two properties make this the
    narrow form of the change:

    * it is **additive** — an already-eligible candidate is scored from exactly
      the same terms as before, because its ``strength`` still comes from
      ``lexical``/``entity``. A dense-only entry joins the ranking; it cannot
      change another candidate's own score, only that candidate's position if it
      happens to outrank it;
    * the floor is **absolute**, not normalized. Admitting on a relative
      threshold would let an unrelated query return its nearest distractor in a
      corpus where every neighbour is somewhat similar.

    A strength term is mandatory rather than cosmetic here. A dense-only
    candidate has coverage 0 and no identifier evidence, so its score would be
    ``0.40 * base`` alone — and with ``dense`` weighted 0.70 against a
    lexical+entity head at 2.15, ``base`` lands near 0.33, putting it at ~0.13,
    *under* the 0.15 noise gate. It would be admitted and then dropped. Its
    strength is therefore the cosine distance above the floor, which is an
    absolute quantity and is below 1.0 unless the match is near-identical.
    """
    informative = [
        c for c in candidates if any(name in c.channels for name in INFORMATIVE_CHANNELS)
    ]
    dense_only: list[Candidate] = []
    dense_only_ids: set[int] = set()
    if dense_only_min_similarity is not None:
        found = [
            c
            for c in candidates
            if not any(name in c.channels for name in INFORMATIVE_CHANNELS)
            and c.channel_scores.get("dense", 0.0) >= dense_only_min_similarity
        ]
        # Best-match first, so the cap keeps the strongest neighbours.
        found.sort(key=lambda c: (-c.channel_scores.get("dense", 0.0), c.memory_id))
        if dense_only_max > 0:
            found = found[:dense_only_max]
        dense_only = found
        dense_only_ids = {c.memory_id for c in found}

    eligible = informative + dense_only
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

        # A dense-only candidate has no informative-channel score to normalize,
        # so its strength is how far its cosine sits above the admission floor.
        # Deliberately not normalized against the pool: dividing by the best
        # similarity would hand every admitted candidate a strength near 1.0 in
        # a corpus where all neighbours are alike, which is the same trap
        # max-normalisation set for the cross-encoder.
        if cand.memory_id in dense_only_ids and dense_only_min_similarity is not None:
            span = 1.0 - dense_only_min_similarity
            if span > 0:
                similarity = cand.channel_scores.get("dense", 0.0)
                strength = max(
                    strength,
                    max(0.0, min(1.0, (similarity - dense_only_min_similarity) / span)),
                )

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

        # The fifth term, off unless ``operative_weight`` is set. Motivated by a
        # measurement (scripts/diagnose_entry_level.py): the chunk that carries
        # the session's score is the one naming a task file only 12.9% of the
        # time, because a read of that file outscores the edit that changed it on
        # every channel -- the paths carry the term weight, the verb carries
        # none. `assemble` already fixes this *inside* a session by promotion;
        # this fixes it in the score, which is what decides which session and
        # which chunk survive the token prefix.
        #
        # It has to be the product, not the bare operative flag. Plain
        # ``_operative_term`` was measured first and moved every number the wrong
        # way, monotonically in the weight (head_is_gold 12.9% -> 12.3 -> 11.7 ->
        # 10.8; gold reaching the payload 33.3% -> 31.1). The reason is that
        # distractor sessions are full of edits too -- of *other* files -- so an
        # unconditioned action term lifts the competition as much as the answer,
        # which is the same failure recorded for promotion applied to every
        # session (ambiguity 0.367 -> 0.567). Requiring the chunk to also name an
        # identifier the question names is what makes it the edit you care about.
        operative = 0.0
        if operative_weight and memory is not None:
            operative = _operative_term(memory.text) * entity_signal

        # Superseded memories stay visible but rank lower: an old fix that a
        # later session replaced is still potentially the useful precedent.
        penalty = 0.65 if (memory is not None and memory.superseded_by) else 1.0

        share = 1.0 - operative_weight
        combined = share * (
            RRF_WEIGHT_IN_FINAL * base
            + COVERAGE_WEIGHT_IN_FINAL * coverage
            + STRENGTH_WEIGHT_IN_FINAL * strength
            + ENTITY_WEIGHT_IN_FINAL * entity_signal
        ) + operative_weight * operative
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


def _line_scores(
    lines: list[str],
    query_terms: Sequence[str],
    operative_weight: float,
) -> list[float]:
    """Score each line by query-term overlap and operative markers."""
    lowered = [line.lower() for line in lines]
    terms = [t for t in query_terms if t]
    scores: list[float] = []
    for i, line in enumerate(lines):
        score = float(sum(1 for term in terms if term in lowered[i]))
        if operative_weight:
            score += operative_weight * _operative_score(line)
        scores.append(score)
    return scores


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

    When operative_weight is set, the algorithm switches to a **multi-span**
    mode: it first extracts every operative block (an operative line plus a
    small amount of surrounding context), then fills the remaining budget with
    the densest non-operative window.  This avoids the situation where a single
    continuous window covers one edit but drops another edit in the same session.

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

    costs = [count_tokens(line) + 1 for line in lines]
    line_scores = _line_scores(lines, query_terms, operative_weight)

    # ------------------------------------------------------------------
    # Fast path: no operative weight -> single best window (unchanged).
    # ------------------------------------------------------------------
    if not operative_weight:
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
                score += line_scores[end]
                end += 1
            if end > start and score > best_score:
                best_score, best_start, best_end = score, start, end

        if best_end <= best_start:
            return truncate_to_tokens(text, budget_tokens), True
        span = "\n".join(lines[best_start:best_end]).strip()
        if not span:
            return truncate_to_tokens(text, budget_tokens), True
        if best_start > 0:
            span = "…\n" + span
        if best_end < len(lines):
            span = span + "\n…"
        return span, True

    # ------------------------------------------------------------------
    # Multi-span path: preserve every operative block, then fill remainder.
    # ------------------------------------------------------------------
    # 1. Identify operative lines and expand each to a small context window.
    #    Context is up to 2 lines above/below, stopping at blank lines or
    #    the next operative line to avoid merging unrelated edits.
    # ------------------------------------------------------------------
    def _is_operative(i: int) -> bool:
        return _operative_score(lines[i]) > 0

    selected = [False] * len(lines)
    total_cost = 0

    for i in range(len(lines)):
        if not _is_operative(i) or selected[i]:
            continue
        # Expand to a small, bounded context around this operative line.
        # Hard limit of 2 lines above/below prevents a long prose paragraph
        # from swallowing the entire budget.
        block_start = max(0, i - 2)
        while block_start > 0 and not _is_operative(block_start - 1):
            if lines[block_start - 1].strip() == "":
                break
            block_start -= 1
        block_start = max(block_start, i - 2)  # enforce 2-line ceiling

        block_end = min(len(lines), i + 3)
        while block_end < len(lines) and not _is_operative(block_end):
            if lines[block_end].strip() == "":
                break
            block_end += 1
        block_end = min(block_end, i + 3)  # enforce 2-line ceiling

        block_cost = sum(costs[j] for j in range(block_start, block_end))
        if total_cost + block_cost > budget_tokens:
            continue

        for j in range(block_start, block_end):
            selected[j] = True
        total_cost += block_cost

    # ------------------------------------------------------------------
    # 2. (Optional) Fill remaining budget with the best non-operative window.
    #    Disabled by default in multi-span mode: once the operative signal is
    #    preserved, padding with prose often adds distractor noise without
    #    raising decisive_present.  The caller can still raise the token budget
    #    if it wants more coverage.
    # ------------------------------------------------------------------
    remaining = budget_tokens - total_cost
    if remaining > 50:  # only fill if there is substantial slack
        best_start = best_end = 0
        best_score = -1.0
        for start in range(len(lines)):
            cost = 0
            score = 0.0
            end = start
            while end < len(lines):
                if selected[end]:
                    end += 1
                    continue
                cost += costs[end]
                if cost > remaining:
                    break
                score += line_scores[end]
                end += 1
            new_lines = sum(1 for j in range(start, end) if not selected[j])
            if new_lines > 0 and score > best_score:
                best_score, best_start, best_end = score, start, end

        if best_end > best_start:
            for j in range(best_start, best_end):
                selected[j] = True

    # ------------------------------------------------------------------
    # 3. Build the span, keeping original line order and marking gaps.
    # ------------------------------------------------------------------
    parts: list[str] = []
    in_gap = False
    for i, line in enumerate(lines):
        if selected[i]:
            if in_gap:
                parts.append("…")
                in_gap = False
            parts.append(line)
        else:
            in_gap = True

    if not parts:
        return truncate_to_tokens(text, budget_tokens), True

    span = "\n".join(parts).strip()
    if not span:
        return truncate_to_tokens(text, budget_tokens), True

    truncated = not all(selected)
    if truncated and not span.startswith("…") and selected[0] is False:
        span = "…\n" + span
    if truncated and not span.endswith("…") and selected[-1] is False:
        span = span + "\n…"
    return span, truncated


def _position_ordered(
    group: list[Candidate], span: tuple[int, int], weight: float
) -> list[Candidate]:
    """Re-order one session's candidates by score nudged toward the trajectory's end.

    Intra-session on purpose. Which sessions are emitted, and in what order, is
    decided upstream by ``final``; this only chooses among the chunks of a session
    that has already been chosen, so a "later is more decisive" prior cannot buy
    a distractor session a slot -- the failure mode that made the same signal
    useless as a scoring term (see ``operative_rank_weight``).

    The tilt is multiplicative and bounded by ``weight``/2, so a candidate cannot
    be promoted past another that scores more than ``weight``/2 above it: with
    0.4, a sibling at 0.60 can overtake one at 0.85 but not one at 0.95.
    """
    lo, hi = span
    if hi <= lo or weight <= 0:
        return group

    def adjusted(cand: Candidate) -> float:
        rel = (cand.memory_id - lo) / (hi - lo)
        return cand.final * (1.0 + weight * (rel - 0.5))

    return sorted(group, key=adjusted, reverse=True)


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

    # Row ids run in message order within a session, so (id - lo) / (hi - lo) is
    # the candidate's position in its own trajectory. One query, not one per
    # session: this runs on every search.
    spans = (
        store.session_span(user_id, order) if settings.evidence_position_weight else {}
    )

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
        # same-repository distractors. Read on the group before any reordering,
        # so the session's score stays its strongest chunk whatever slot order
        # the intra-session levers choose below.
        if group[0].final < settings.min_evidence_score and len(
            items
        ) >= settings.min_evidence_count:
            break

        if settings.evidence_position_weight:
            span = spans.get(session_id)
            if span is not None:
                group = _position_ordered(
                    group, span, settings.evidence_position_weight
                )

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

            # Confidence-aware multi-span: the top-ranked session(s) use the
            # multi-span treatment so decisive operative blocks survive
            # truncation; later sessions revert to single-window selection so
            # distractor operative lines do not inflate ambiguity.
            operative_weight = (
                settings.evidence_operative_weight
                if session_index < 1
                else 0.0
            )
            content, truncated = _select_span(
                body_source,
                budget_for_item,
                plan.keywords,
                operative_weight,
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
