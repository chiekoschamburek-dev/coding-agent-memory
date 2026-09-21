"""Retrieval metrics for the local proxy benchmark.

CAMBench Coding is not public, so iteration needs a local proxy evaluation. The
metrics here are the ones that actually track the platform's scoring, given that
the answer model receives a token-counted prefix of our ranked output:

* ``recall_at_k``  — was the needed evidence in the prefix at all (a hard
  ceiling: if it is absent, no answer model can use it);
* ``ndcg_at_k``    — is it ranked *high*, which is what determines whether it
  survives prefix truncation;
* ``mrr``          — how far down the first useful item sits;
* ``precision_at_k`` — how much of the prefix is signal rather than same-repo
  noise, i.e. how much of the context budget is wasted.

All are query-averaged. Relevant ids are supplied per query by the benchmark
builder; this module knows nothing about how relevance was determined.

Two accounting units
--------------------
``/search`` returns ``data[]`` *memory entries*, and several entries come from the
same session (roughly two per session on this corpus), while relevance is
labelled per session. The platform's ``top_k`` and token prefix both cut the
entry list, so scoring only the session-collapsed list reports a recall the
answer model cannot see. ``evaluate`` therefore scores any ranked sequence under
each unit: pass ``key="ranked"`` for the session view and ``key="ranked_items"``
for the entry view the platform actually truncates.

Two of the three hit-based metrics count *distinct* sessions, so an entry list
padded with repeats of one session cannot inflate them: ``recall_at_k`` takes a
set intersection, and ``ndcg_at_k`` credits a document only on first appearance.
``precision_at_k`` deliberately does not — it counts slots, because a repeated
session consuming three of them is a real cost to the budget, not a free hit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the window's *slots* holding evidence from a relevant session.

    Counted over the list, not over distinct ids. Three entries from one relevant
    session occupy three slots, and how much of the budget a repeated session
    consumes is precisely what this metric exists to show. Deduplicating here
    would silently turn it into a different quantity — the density of *distinct*
    relevant sessions — and would score a payload that spends two thirds of its
    budget on repeats as though those slots were free. On the collapsed session
    list the two definitions coincide, so this only affects the ``item_`` view.
    """
    if k <= 0:
        return 0.0
    window = ranked[:k]
    if not window:
        return 0.0
    return sum(1 for doc_id in window if doc_id in relevant) / len(window)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    def dcg(items: Iterable[str]) -> float:
        total = 0.0
        seen: set[str] = set()
        for rank, doc_id in enumerate(items, start=1):
            gain = 1.0 if doc_id in relevant and doc_id not in seen else 0.0
            seen.add(doc_id)
            total += gain / math.log2(rank + 1)
        return total

    actual = dcg(ranked[:k])
    ideal = dcg(list(relevant)[:k])
    return actual / ideal if ideal > 0 else 0.0


def evaluate(
    results: dict[str, dict],
    *,
    ks: Sequence[int] = (10, 100),
    key: str = "ranked",
    prefix: str = "",
) -> dict[str, float]:
    """Aggregate metrics over a query set.

    ``results`` maps query_id to ``{"ranked": [doc_id, ...], "relevant": {...}}``.
    ``key`` selects which ranked sequence to score and ``prefix`` namespaces the
    output keys, so the session view and the memory-entry view of the same run
    can sit in one result dict. Response-level rates are only reported for the
    unprefixed pass because they describe the response, not the ranking unit.
    """
    if not results:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        out[f"{prefix}recall@{k}"] = _mean(
            recall_at_k(r[key], set(r["relevant"]), k) for r in results.values()
        )
        out[f"{prefix}ndcg@{k}"] = _mean(
            ndcg_at_k(r[key], set(r["relevant"]), k) for r in results.values()
        )
        out[f"{prefix}precision@{k}"] = _mean(
            precision_at_k(r[key], set(r["relevant"]), k) for r in results.values()
        )
    out[f"{prefix}mrr"] = _mean(
        reciprocal_rank(r[key], set(r["relevant"])) for r in results.values()
    )
    if prefix:
        return out
    detected = sum(1 for r in results.values() if r[key])
    out["detection_rate"] = detected / len(results)
    out["avg_returned"] = _mean(len(r[key]) for r in results.values())
    return out


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def compare(baseline: dict[str, float], candidate: dict[str, float]) -> str:
    """Render a side-by-side ablation table."""
    keys = sorted(set(baseline) | set(candidate))
    lines = [f"{'metric':<18}{'baseline':>10}{'candidate':>12}{'delta':>10}"]
    lines.append("-" * 50)
    for key in keys:
        b = baseline.get(key)
        c = candidate.get(key)
        b_str = f"{b:.4f}" if isinstance(b, float) else "-"
        c_str = f"{c:.4f}" if isinstance(c, float) else "-"
        if isinstance(b, float) and isinstance(c, float):
            d = f"{c - b:+.4f}"
        else:
            d = "-"
        lines.append(f"{key:<18}{b_str:>10}{c_str:>12}{d:>10}")
    return "\n".join(lines)
