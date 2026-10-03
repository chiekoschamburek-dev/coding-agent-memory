"""Experiment: entry-level ordering vs session-major packing.

Question
--------
``assemble`` currently emits memories *session by session*: it groups the
globally-scored candidate list by ``session_id``, walks the sessions in the
order their best entry ranks, and takes up to ``max_evidence_per_session``
entries from each. The effect is that a session's weaker entries are emitted
before another session's stronger ones, because the session block travels as a
unit.

This experiment asks whether emitting **entries in global score order**, with
the per-session cap kept merely as a quota, does better on the metrics the
platform actually cuts -- ``top_k`` counts ``data[]`` entries, and the answer
model reads a token-counted prefix of that entry list.

Design
------
Only the assembler changes. Recall, fusion, scoring and reranking are shared,
because the candidate list handed to the assembler is identical in every arm:
``score_candidates`` already returns entries sorted by ``final`` descending, so
the four arms differ solely in how they walk that same list.

    S0  session-major, cap 5   (shipped behaviour) - baseline
    S1  entry order, no cap    (pure entry ordering; exposes near-duplicate crowding)
    S2  entry order, cap 5     (order and quota decoupled - the hypothesis)
    S3  entry order, cap 1     (maximum session diversity)

    S0 vs S2  -> pure ordering effect (cap held at 5)
    S2 vs S1  -> pure quota effect    (order held at entry-level)
    S2 vs S3  -> cap gradient

Under a shared cap, S2 is a global greedy over ``final`` while S0 is a greedy
over session blocks, so S2 >= S0 holds by construction. What is being measured
is the *size* and *significance* of that gap, not its direction.

Nothing under ``src/`` is modified: the assembler is swapped at runtime by
patching the module-level name ``codemem.search.service.assemble``.

Usage::

    python scripts/exp_entry_order.py --limit 15          # smoke
    python scripts/exp_entry_order.py                     # full, 89 queries
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))  # noqa: E402 - harness path, like run_benchmark

from codemem.core.tokens import count_tokens  # noqa: E402
from codemem.search.evidence import (  # noqa: E402
    EvidenceItem,
    _iso_from_ms,
    _select_span,
    assemble as assemble_session_major,
)
from metrics import (  # noqa: E402
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# --------------------------------------------------------------------------
# The entry-order assembler. Every behaviour other than the walk order and the
# per-session quota is imported from the shipped assembler rather than
# reimplemented, so span selection, token budgeting, text dedup, the noise gate
# and the strictly-decreasing score clamp stay byte-for-byte comparable.
# --------------------------------------------------------------------------


def assemble_entry_major(
    settings: Any,
    store: Any,
    user_id: str,
    plan: Any,
    scored: Sequence[Any],
    memories: dict[int, Any],
    *,
    top_k: int,
    cap: int = 3,
) -> list[EvidenceItem]:
    """Walk candidates in global score order, taking at most ``cap`` per session.

    ``cap <= 0`` means no per-session quota at all, which is the pure "sort by
    evidence unit" variant and the arm that shows whether near-duplicate entries
    from one session crowd the window out on their own.

    Two deliberate differences from the session-major walk, one of which is
    the thing under test rather than an accident:

    * the noise gate is applied the way the shipped assembler applies it — to
      a session's *best* evidence, i.e. only when admitting the session's
      first entry (its head in the descending list), and the walk terminates
      on the first below-gate session head. Gating every candidate instead
      ends the walk on a weak non-head chunk of an already-admitted session
      while later sessions' heads — which rank above that chunk — would still
      have cleared the gate: a quieter payload that measures the gate, not
      the ordering. (First version gated every candidate; S2 came back at 4.1
      entries per query against the baseline's 9.2, which is how this was
      caught.);
    * ``_role_ordered`` (promoting a session's operative chunk) is not applied.
      That promotion is itself an intra-session intervention, and folding it in
      would confound the ordering effect this experiment isolates.
    """
    if not scored or top_k <= 0:
        return []

    # Spans come from the chunk text when available, exactly as the shipped
    # assembler does, so both arms select windows over identical source text.
    chunk_ids = [
        memories[c.memory_id].chunk_id
        for c in scored
        if c.memory_id in memories and memories[c.memory_id].chunk_id is not None
    ]
    chunk_map = store.fetch_chunks(user_id, chunk_ids)
    source_of: dict[int, str] = {}
    for cand in scored:
        memory = memories.get(cand.memory_id)
        if memory is None:
            continue
        chunk_id = memory.chunk_id
        source_of[cand.memory_id] = (
            chunk_map[chunk_id].text if chunk_id in chunk_map else memory.text
        )

    budget = settings.evidence_budget_tokens
    max_sessions = settings.evidence_max_sessions  # 0 = unlimited, as shipped
    used = 0
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    taken_per_session: Counter[str] = Counter()
    sessions_used = 0
    # Emission order is the relevance order we assert, so scores are clamped to
    # it: returned order and returned scores can never disagree.
    ceiling = float("inf")

    for cand in scored:
        if len(items) >= top_k:
            break
        memory = memories.get(cand.memory_id)
        if memory is None:
            continue

        session_id = memory.session_id
        new_session = taken_per_session[session_id] == 0
        # The session limit stops *admission*, not the walk. Breaking here —
        # on the first candidate of a third session — strands the entries the
        # two admitted sessions still have above the quota and rank below the
        # third session's head (second version: S2 came back at 5.2 entries
        # against the baseline's 9.2). The shipped assembler never has this
        # problem because it drains each admitted session's block before
        # moving to the next session, so the entry walk skips further sessions
        # and keeps harvesting.
        if new_session and max_sessions and sessions_used >= max_sessions:
            continue
        if cap > 0 and taken_per_session[session_id] >= cap:
            continue  # quota spent; keep scanning for another session's entries

        # Absolute evidence gate, read on the session's *best* evidence: the
        # first candidate seen for a session IS its head in this descending
        # walk, so the gate fires when admitting the session and the walk ends
        # on the first below-gate session head — the shipped semantics. A weak
        # non-head chunk of an admitted session must neither be taken (the
        # gate-break below would end the walk there, silently dropping later
        # sessions whose heads cleared the gate) nor end the walk.
        if (
            new_session
            and cand.final < settings.min_evidence_score
            and len(items) >= settings.min_evidence_count
        ):
            break

        body_source = source_of.get(cand.memory_id, memory.text)
        full_form = len(items) < settings.evidence_full_count
        budget_for_item = (
            settings.evidence_item_tokens if full_form else settings.evidence_ptr_tokens
        )
        content, truncated = _select_span(
            body_source,
            budget_for_item,
            plan.keywords,
            settings.evidence_operative_weight,
        )
        stamped = _iso_from_ms(memory.ts)

        # Repeated text across sessions adds no evidence but consumes the
        # answer model's context.
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
                created_at=stamped or memory.created_at,
                tokens=item_tokens,
                truncated=truncated,
                superseded=memory.superseded_by is not None,
            )
        )
        taken_per_session[session_id] += 1
        if taken_per_session[session_id] == 1:
            sessions_used += 1

    # Strictly decreasing by more than the rounding can absorb: a 1e-7 relative
    # nudge would round back onto its neighbour and yield equal scores.
    for idx, item in enumerate(items):
        item.score = round(max(item.score - idx * 1e-5, 1e-6), 6)
    return items


# cap = 0 means "no quota"; None selects the shipped assembler untouched.
# The per-session cap follows the shipped default, 5 as of 2026-09-25 (the
# multi-span decision): S0 vs S2 is a pure ordering contrast only while both
# arms spend the same quota, so the entry-order hypothesis is tested at 5, not
# at the cap-3 default this file was first written against.
STRATEGIES: dict[str, Callable[..., Any] | None] = {
    "S0_session_cap5": None,
    "S1_entry_nocap": partial(assemble_entry_major, cap=0),
    "S2_entry_cap5": partial(assemble_entry_major, cap=5),
    "S3_entry_cap1": partial(assemble_entry_major, cap=1),
}

BASELINE = "S0_session_cap5"

# The platform counts entries, so the item_ view is decision-relevant. Session
# rows are reported alongside to show whether an item-view gain is bought by
# silently narrowing answer coverage.
PER_ENTRY_METRICS: dict[str, Callable[[dict], float]] = {
    "item_recall@10": lambda r: recall_at_k(r["ranked_items"], r["relevant"], 10),
    "item_ndcg@10": lambda r: ndcg_at_k(r["ranked_items"], r["relevant"], 10),
    "item_precision@10": lambda r: precision_at_k(r["ranked_items"], r["relevant"], 10),
    "item_mrr": lambda r: reciprocal_rank(r["ranked_items"], r["relevant"]),
    "item_recall@100": lambda r: recall_at_k(r["ranked_items"], r["relevant"], 100),
}
PER_SESSION_METRICS: dict[str, Callable[[dict], float]] = {
    "recall@10": lambda r: recall_at_k(r["ranked"], r["relevant"], 10),
    "ndcg@10": lambda r: ndcg_at_k(r["ranked"], r["relevant"], 10),
    "mrr": lambda r: reciprocal_rank(r["ranked"], r["relevant"]),
}


# ------------------------------------------------------------------ stats --


def paired_bootstrap(
    before: dict[str, float],
    after: dict[str, float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Paired bootstrap over queries: delta, 95% CI, two-sided p.

    The project judges every change this way (``eval/README.md`` retires
    variants at p ~ 0.25 rather than shipping them), so the experiment reports
    the same statistics rather than raw deltas that could be host noise. Only
    queries present in both arms are paired.
    """
    keys = sorted(set(before) & set(after))
    va = [before[k] for k in keys]
    vb = [after[k] for k in keys]
    if not keys:
        return {"delta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p": float("nan"), "n": 0}

    delta = statistics.fmean(vb) - statistics.fmean(va)
    n = len(keys)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistics.fmean(vb[i] for i in idx)
                     - statistics.fmean(va[i] for i in idx))
    diffs.sort()

    # The two arms can be identical on a metric (this is expected for the
    # session view: every arm consumes the same candidate pool, so the set of
    # sessions in the payload does not move). A constant-zero difference is the
    # *weakest* possible evidence, so report p = 1.0. Letting the two-sided
    # formula see it turns "no difference at all" into p = 0, i.e. the most
    # "significant" row in the table -- the opposite of the truth.
    if abs(diffs[0]) < 1e-12 and abs(diffs[-1]) < 1e-12:
        return {"delta": delta, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0,
                "n": n, "queries_moved": 0}

    def quantile(q: float) -> float:
        pos = min(max(int(q * resamples), 0), resamples - 1)
        return diffs[pos]

    # Two-sided p directly from the resampled difference distribution.
    le = sum(1 for d in diffs if d <= 0)
    p = 2 * min(le, resamples - le) / resamples
    return {
        "delta": delta,
        "ci_low": quantile(0.025),
        "ci_high": quantile(0.975),
        "p": min(1.0, p),
        "n": n,
        "queries_moved": sum(1 for a, b in zip(va, vb) if abs(b - a) > 1e-12),
    }


def _sig(p: float) -> str:
    return "yes" if p < 0.05 else "no"


# -------------------------------------------------------------------- run --


def load_benchmark(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def user_for_repo(data: dict, repo: str) -> str:
    """The benchmark isolates memory per repository (``bench:<repo>``)."""
    for memory in data["memories"]:
        if memory["repo"] == repo:
            return memory["user_id"]
    return f"bench:{repo}"


def run(
    data: dict,
    *,
    top_k: int,
    limit: int | None,
    ks: tuple[int, ...],
    resamples: int,
    seed: int,
    quiet: bool,
) -> dict:
    import logging
    import tempfile

    from fastapi.testclient import TestClient

    import codemem.search.service as service_module
    from codemem.api.app import create_app
    from codemem.core.config import Settings

    logging.disable(logging.WARNING)

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    app = create_app(settings)

    memories = data["memories"]
    queries = [q for q in data["queries"] if q["relevant"]]
    if limit:
        queries = queries[:limit]

    started = time.monotonic()
    with TestClient(app) as client:
        store = app.state.container.store

        # ---- Add once; every arm then reads the same indexed corpus -------
        added = skipped = 0
        for memory in memories:
            payload = {
                "request_id": f"bench:{memory['id']}",
                "user_id": memory["user_id"],
                "session_id": memory["session_id"],
                "messages": memory["messages"],
            }
            response = client.post("/add", json=payload)
            if response.status_code != 200:
                skipped += 1
                if not quiet:
                    print(f"  add failed for {memory['id']}: {response.status_code}")
                continue
            added += 1
        add_seconds = time.monotonic() - started

        # ---- memory_id -> session_id, the bridge to the file-overlap gold --
        id_to_session: dict[str, dict[str, str]] = {}
        with store._read() as conn:  # noqa: SLF001 - eval harness
            for row in conn.execute("SELECT id, user_id, session_id FROM memory"):
                id_to_session.setdefault(row["user_id"], {})[
                    f"mem_{row['id']}"
                ] = row["session_id"]

        channels = app.state.container.embedder, app.state.container.reranker
        dense_available = bool(channels[0] and channels[0].available)
        rerank_available = bool(channels[1] and channels[1].available)
        if not quiet:
            print(f"  dense available={dense_available} rerank available={rerank_available}")

        # ---- Search once per arm, swapping only the assembler -------------
        original_assemble = service_module.assemble
        all_results: dict[str, dict[str, dict]] = {}
        timings: dict[str, float] = {}

        try:
            for name, replacement in STRATEGIES.items():
                service_module.assemble = replacement or original_assemble
                arm_started = time.monotonic()
                results: dict[str, dict] = {}
                empties = 0
                for query in queries:
                    user_id = user_for_repo(data, query["repo"])
                    response = client.post(
                        "/search",
                        json={"query": query["query"], "user_id": user_id,
                              "top_k": top_k},
                    )
                    items = response.json().get("data", [])
                    if not items:
                        empties += 1
                    lookup = id_to_session.get(user_id, {})
                    # Session view: first-appearance order, several entries of
                    # one session are one piece of evidence.
                    ranked: list[str] = []
                    for item in items:
                        # The id carries a "_superseded" suffix; strip it so it
                        # resolves in the memory_id -> session map.
                        raw_id = item["id"].split("_superseded")[0]
                        session = lookup.get(raw_id)
                        if session and session not in ranked:
                            ranked.append(session)
                    # Entry view: what top_k and the token prefix actually cut.
                    ranked_items = [
                        lookup.get(item["id"].split("_superseded")[0]) for item in items
                    ]
                    results[query["query_id"]] = {
                        "ranked": ranked,
                        "ranked_items": ranked_items,
                        "relevant": set(query["relevant"]),
                        "repo": query["repo"],
                    }
                all_results[name] = results
                timings[name] = round(time.monotonic() - arm_started, 1)
                if not quiet:
                    mean_items = statistics.fmean(
                        len(r["ranked_items"]) for r in results.values()
                    )
                    print(f"  [{name}] {len(results)} queries, "
                          f"{mean_items:.1f} entries avg, {empties} empty, "
                          f"{timings[name]}s")
        finally:
            service_module.assemble = original_assemble

    return {
        "results": all_results,
        "counts": {"memories_added": added, "memories_skipped": skipped,
                   "queries": len(queries)},
        "add_seconds": round(add_seconds, 1),
        "timings": timings,
        "channels": {"dense_available": dense_available,
                     "rerank_available": rerank_available},
    }


def _aggregate(
    results: dict[str, dict], ks: tuple[int, ...]
) -> dict[str, float]:
    """Both accounting units plus the packing shape, mirroring run_benchmark."""
    metrics = evaluate(results, ks=ks)
    metrics.update(evaluate(results, ks=ks, key="ranked_items", prefix="item_"))
    avg_items = statistics.fmean(len(r["ranked_items"]) for r in results.values())
    avg_sessions = statistics.fmean(len(r["ranked"]) for r in results.values())
    metrics["avg_returned"] = avg_items
    metrics["avg_sessions_returned"] = avg_sessions
    metrics["items_per_session"] = (
        round(avg_items / avg_sessions, 2) if avg_sessions else 0.0
    )
    # How many distinct sessions the first ten *entry slots* span -- the direct
    # measure of the crowding the per-session cap exists to prevent.
    metrics["sessions_in_top10_slots"] = statistics.fmean(
        len({s for s in r["ranked_items"][:10] if s}) for r in results.values()
    )
    metrics["empty_rate"] = (
        sum(1 for r in results.values() if not r["ranked_items"]) / len(results)
        if results
        else 0.0
    )
    return metrics


def _per_query(
    results: dict[str, dict], specs: dict[str, Callable[[dict], float]]
) -> dict[str, dict[str, float]]:
    return {
        name: {qid: fn(res) for qid, res in results.items()} for name, fn in specs.items()
    }


def report(
    per_arm: dict[str, dict[str, dict[str, float]]],
    aggregates: dict[str, dict[str, float]],
    *,
    comparisons: Iterable[tuple[str, str]],
    resamples: int,
    seed: int,
) -> dict:
    head = f"{'metric':<24}" + "".join(f"{n.split('_')[1][:9]:>11}" for n in aggregates)
    print()
    print("=" * len(head))
    print("Entry-level ordering vs session-major packing (paired, 89 queries max)")
    print("=" * len(head))
    print(head)
    print("-" * len(head))

    ordered = [
        "recall@10", "ndcg@10", "mrr",
        "item_recall@10", "item_ndcg@10", "item_precision@10", "item_mrr",
        "item_recall@100",
        "avg_returned", "avg_sessions_returned", "items_per_session",
        "sessions_in_top10_slots", "empty_rate",
    ]
    for metric in ordered:
        row = ""
        for name in aggregates:
            value = aggregates[name].get(metric)
            row += f"{value:>11.4f}" if isinstance(value, float) else f"{'-':>11}"
        print(f"{metric:<24}{row}")
    print("-" * len(head))
    caps = {"S0_session_cap5": 5, "S1_entry_nocap": 0,
            "S2_entry_cap5": 5, "S3_entry_cap1": 1}
    for name in aggregates:
        kind = "entry-order" if STRATEGIES[name] else "session-major"
        cap = caps[name]
        print(f"  {name:<22}= {kind}, cap={cap if cap else 'unlimited'}")

    print()
    print(f"Paired bootstrap, {resamples} resamples, seed {seed} (p<0.05 is the "
          f"project's bar)")
    head = (f"{'comparison':<35}{'metric':<20}{'delta':>9}{'95% CI':>20}"
            f"{'p':>8}{'sig':>5}{'moved':>7}")
    print(head)
    print("-" * len(head))

    stats: dict[str, dict] = {}
    for left, right in comparisons:
        for spec_name, specs in (("entry", PER_ENTRY_METRICS),
                                 ("session", PER_SESSION_METRICS)):
            for metric in specs:
                if metric not in ("item_recall@10", "item_ndcg@10",
                                  "item_precision@10", "item_mrr", "mrr"):
                    continue
                out = paired_bootstrap(
                    per_arm[left][metric], per_arm[right][metric],
                    resamples=resamples, seed=seed,
                )
                label = f"{right} - {left}"
                stats[f"{label}::{metric}"] = out
                ci = f"[{out['ci_low']:+.4f},{out['ci_high']:+.4f}]"
                print(f"{label:<35}{metric:<20}{out['delta']:>+9.4f}"
                      f"{ci:>20}{out['p']:>8.3f}{_sig(out['p']):>5}"
                      f"{out['queries_moved']:>7}")
    return stats


def _describe_path(channels: dict[str, bool]) -> str:
    """Which retrieval path this host actually ran, so results are read right."""
    dense, rerank = channels["dense_available"], channels["rerank_available"]
    if dense and rerank:
        return "P2c (dense + rerank live - the shipped default configuration)"
    if not dense and not rerank:
        return "P1 (dense and rerank unavailable; deterministic lexical+entity)"
    return f"mixed (dense={dense}, rerank={rerank}) - compare deltas only"


def _pool_identity(all_results: dict[str, dict], base: str = BASELINE) -> dict[str, float]:
    """How often each arm returns exactly the baseline's *set* of sessions.

    This is the pairing check. Every arm walks the same scored candidate list,
    so ideally the sessions present in the payload never differ and the whole
    delta comes from ordering. Where the sets do differ -- the unlimited-cap arm
    can exhaust the token budget before reaching later sessions -- those queries
    contribute budget effects as well, and the number here says how many.
    """
    base_sets = {qid: frozenset(r["ranked"]) for qid, r in all_results[base].items()}
    out: dict[str, float] = {}
    for name, results in all_results.items():
        if name == base or not results:
            continue
        same = sum(
            1 for qid, res in results.items()
            if frozenset(res["ranked"]) == base_sets.get(qid, frozenset())
        )
        out[name] = same / len(results)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "eval" / "data" / "benchmark.json")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ks", default="10,100")
    parser.add_argument("--limit", type=int, default=None, help="query subset")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "eval" / "results" / "exp_entry_order.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(f"error: {args.data} not found; run eval/build_benchmark.py first",
              file=sys.stderr)
        return 2

    ks = tuple(int(k) for k in args.ks.split(",") if k.strip())
    data = load_benchmark(args.data)
    out = run(
        data, top_k=args.top_k, limit=args.limit, ks=ks,
        resamples=args.resamples, seed=args.seed, quiet=args.quiet,
    )

    per_arm: dict[str, dict[str, dict[str, float]]] = {}
    for name, results in out["results"].items():
        merged = dict(_per_query(results, PER_ENTRY_METRICS))
        merged.update(_per_query(results, PER_SESSION_METRICS))
        per_arm[name] = merged

    aggregates = {name: _aggregate(results, ks)
                  for name, results in out["results"].items()}

    comparisons = [
        (BASELINE, "S2_entry_cap5"),   # ordering effect, cap held at 5
        ("S2_entry_cap5", "S1_entry_nocap"),  # quota effect, order held
        ("S2_entry_cap5", "S3_entry_cap1"),   # cap gradient
        (BASELINE, "S1_entry_nocap"),   # both changes at once
    ]
    stats = report(per_arm, aggregates, comparisons=comparisons,
                   resamples=args.resamples, seed=args.seed)

    pool_identity = _pool_identity(out["results"])
    print()
    print("Pairing check - fraction of queries returning the baseline's session set:")
    for name, share in pool_identity.items():
        print(f"  {name:<35}{share:>8.1%}")
    print("  (below 100% means that arm also changes *which* sessions fit, not "
          "only their order)")

    payload = {
        "experiment": "entry-level ordering vs session-major packing",
        "note": ("Only the assembler differs; recall/fusion/scoring/reranking are "
                 "shared because every arm walks the same scored candidate list. "
                 "src/ was not modified - the assembler is patched at runtime."),
        "config": {
            "top_k": args.top_k, "ks": list(ks), "limit": args.limit,
            "resamples": args.resamples, "seed": args.seed,
            "max_evidence_per_session": "shipped default 5 (S0/S2 baseline; 2026-09-25 multi-span decision)",
            "evidence_operative_promotion": "not applied in S1/S2/S3 (isolated variable)",
        },
        "environment": {
            "channels": out["channels"],
            "path": _describe_path(out["channels"]),
            "warning": ("Read the paired deltas, not the absolute values: this is a "
                        "proxy benchmark with file-overlap ground truth."),
        },
        "counts": out["counts"],
        "add_seconds": out["add_seconds"],
        "arm_seconds": out["timings"],
        "metrics": aggregates,
        "paired_bootstrap": stats,
        "pool_identity_vs_baseline": pool_identity,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
