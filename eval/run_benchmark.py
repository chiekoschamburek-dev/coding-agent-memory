"""Run the proxy retrieval benchmark against the service.

Runs in-process through the real HTTP contract (so schema, auth, and
``top_k`` handling are exercised) and maps returned memory ids back to their
source session through the store, which is how the file-overlap ground truth is
compared.

Metrics and their meaning for this competition
-----------------------------------------------
The platform feeds the answer model a token-counted *prefix* of our ranked
output, so:

* ``recall@k``  — is the needed evidence in the prefix at all? If not, no answer
  model can use it. This is a hard ceiling.
* ``ndcg@k``    — is it ranked high enough to survive prefix truncation?
* ``mrr``       — how far down the first useful memory sits.
* ``precision@k`` — how much of the prefix is signal rather than same-repo noise.
* ``empty``     — fraction of queries returning nothing. High is not automatically
  bad (the noise gate is meant to abstain), but it is a ceiling on recall.

Each ranking metric is reported twice, because the store returns several memory
entries per session and only the ``item_`` rows reflect what a ``top_k`` cut or a
token prefix actually delivers; the unprefixed rows collapse entries onto their
session. See ``metrics`` for why both are needed.

Usage::

    python eval/run_benchmark.py --data eval/data/benchmark.json
    python eval/run_benchmark.py --data ... --top-k 100 --limit 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

from metrics import evaluate  # noqa: E402


def load_benchmark(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run(
    data: dict,
    *,
    top_k: int,
    limit: int | None,
    ks: tuple[int, ...],
    quiet: bool,
    overrides: dict | None = None,
) -> dict:
    from fastapi.testclient import TestClient

    from codemem.api.app import create_app
    from codemem.core.config import Settings

    import logging
    import tempfile

    logging.disable(logging.WARNING)

    settings = Settings(data_dir=Path(tempfile.mkdtemp()), **(overrides or {}))
    app = create_app(settings)

    memories = data["memories"]
    queries = [q for q in data["queries"] if q["relevant"]]
    if limit:
        queries = queries[:limit]

    t0 = time.monotonic()
    with TestClient(app) as client:
        store = app.state.container.store

        # ---- Add every session -------------------------------------------
        added = 0
        skipped = 0
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
        add_seconds = time.monotonic() - t0

        # ---- Map memory_id -> session_id per user ------------------------
        mapping: dict[str, dict[str, str]] = {}
        with store._read() as conn:  # noqa: SLF001 - eval harness
            for row in conn.execute("SELECT id, user_id, session_id FROM memory"):
                mapping.setdefault(row["user_id"], {})[
                    f"mem_{row['id']}"
                ] = row["session_id"]

        # ---- Run queries -------------------------------------------------
        results: dict[str, dict] = {}
        empties = 0
        search_times: list[float] = []

        for query in queries:
            user_id = _user_for_repo(data, query["repo"])
            started = time.monotonic()
            response = client.post(
                "/search",
                json={"query": query["query"], "user_id": user_id, "top_k": top_k},
            )
            search_times.append(time.monotonic() - started)
            items = response.json().get("data", [])
            if not items:
                empties += 1
            lookup = mapping.get(user_id, {})
            # Collapse to sessions, keeping first-appearance order: several
            # chunks of one session are one piece of evidence.
            ranked: list[str] = []
            for item in items:
                session = lookup.get(item["id"])
                if session and session not in ranked:
                    ranked.append(session)
            # The platform's unit is the memory entry, not the session: `top_k`
            # counts `data[]` items and those items are what reach Answer, in
            # order. Keeping the un-collapsed sequence lets the two accounting
            # units be compared instead of silently assuming they agree.
            ranked_items = [lookup.get(item["id"]) for item in items]
            results[query["query_id"]] = {
                "ranked": ranked,
                "ranked_items": ranked_items,
                "relevant": set(query["relevant"]),
                "repo": query["repo"],
            }

    metrics = evaluate(results, ks=ks)
    metrics.update(evaluate(results, ks=ks, key="ranked_items", prefix="item_"))
    metrics["empty_rate"] = empties / len(queries) if queries else 0.0
    avg_items = statistics.fmean(len(r["ranked_items"]) for r in results.values())
    avg_sessions = statistics.fmean(len(r["ranked"]) for r in results.values())
    metrics["avg_returned"] = avg_items
    metrics["avg_sessions_returned"] = avg_sessions
    metrics["items_per_session"] = (
        round(avg_items / avg_sessions, 2) if avg_sessions else 0.0
    )
    metrics["add_seconds"] = round(add_seconds, 1)
    metrics["search_mean_ms"] = (
        round(statistics.fmean(search_times) * 1000, 1) if search_times else 0.0
    )

    random_baseline = _random_baseline(data, queries, ks)
    return {
        "metrics": metrics,
        "random_baseline": random_baseline,
        "results": results,
        "counts": {
            "memories_added": added,
            "memories_skipped": skipped,
            "queries": len(queries),
        },
    }


def _user_for_repo(data: dict, repo: str) -> str:
    """The benchmark isolates memory per repository (``bench:<repo>``)."""
    for memory in data["memories"]:
        if memory["repo"] == repo:
            return memory["user_id"]
    return f"bench:{repo}"


def _random_baseline(data: dict, queries: list[dict], ks: tuple[int, ...]) -> dict:
    """Expected metrics for a random ranking over the same pools.

    Gives the numbers a non-trivial retrieval signal must beat; without it a
    recall figure is unreadable.
    """
    import random

    pool: dict[str, list[str]] = {}
    for memory in data["memories"]:
        pool.setdefault(memory["user_id"], []).append(memory["session_id"])

    rng = random.Random(20260919)
    results: dict[str, dict] = {}
    for query in queries:
        user_id = _user_for_repo(data, query["repo"])
        candidates = list(pool.get(user_id, []))
        rng.shuffle(candidates)
        results[query["query_id"]] = {
            "ranked": candidates,
            "relevant": set(query["relevant"]),
        }
    return evaluate(results, ks=ks)


def report(out: dict, *, ks: tuple[int, ...], meta: dict) -> None:
    metrics = out["metrics"]
    baseline = out["random_baseline"]

    # A random ranking across the same pools cannot score zero unless the
    # comparison itself is broken (e.g. mismatched id namespaces). Refuse to
    # print plausible-looking numbers when the harness is suspect.
    if baseline.get("mrr", 0.0) == 0.0:
        print(
            "\nHARNESS ERROR: the random baseline scored exactly zero, which is "
            "impossible for a non-empty pool. The relevance ids and the returned "
            "ids are almost certainly in different namespaces, so every metric "
            "below is meaningless. Fix the harness before drawing conclusions.",
            file=sys.stderr,
        )

    print()
    print("=" * 68)
    print("Proxy retrieval benchmark (SWEContextBench) — NOT the scored suite")
    print("=" * 68)
    print(f"memories added : {out['counts']['memories_added']} "
          f"(skipped {out['counts']['memories_skipped']})")
    print(f"queries scored : {out['counts']['queries']}")
    print(f"add time       : {out['metrics']['add_seconds']}s")
    print(f"search latency : {metrics['search_mean_ms']}ms mean")
    print()
    header = f"{'metric':<22}{'random':>10}{'codemem':>12}{'lift':>10}"
    print(header)
    print("-" * len(header))
    for key in sorted(metrics):
        if key in ("add_seconds", "search_mean_ms"):
            continue
        value = metrics[key]
        base = baseline.get(key)
        if isinstance(base, float):
            lift = f"{value - base:+.4f}" if isinstance(value, float) else "-"
            print(f"{key:<22}{base:>10.4f}{value:>12.4f}{lift:>10}")
        else:
            shown = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"{key:<22}{'-':>10}{shown:>12}{'-':>10}")

    print()
    for line in _wrap(
        "Unprefixed rows collapse data[] entries onto their source session; "
        "item_ rows score the entry list itself, which is what top_k and the "
        "answer model's token prefix cut. Both count distinct sessions, so "
        "repeat chunks of one session never earn extra credit. The random "
        "baseline is a session ordering, so it has no item_ counterpart.",
        64,
    ):
        print(f"  {line}")

    print()
    print("relevance definition (our proxy):")
    for line in _wrap(meta.get("relevance_definition", ""), 64):
        print(f"  {line}")
    print("caveat:")
    for line in _wrap(meta.get("relevance_caveats", ""), 64):
        print(f"  {line}")


def _wrap(text: str, width: int) -> list[str]:
    words = (text or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("eval/data/benchmark.json")
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ks", default="10,100")
    parser.add_argument("--limit", type=int, default=None, help="query subset")
    parser.add_argument("--out", type=Path, default=None, help="dump metrics json")
    parser.add_argument("--dump-per-query", type=Path, default=None)
    parser.add_argument("--noise-gate", type=float, default=None,
                        help="override min_evidence_score (0 disables the gate)")
    parser.add_argument("--budget-tokens", type=int, default=None,
                        help="override evidence token budget")
    parser.add_argument("--full-count", type=int, default=None,
                        help="override how many items use the full evidence form")
    parser.add_argument("--recall-limit", type=int, default=None,
                        help="max items the assembler may return")
    parser.add_argument("--item-tokens", type=int, default=None,
                        help="cap for items rendered in full form")
    parser.add_argument("--cap", type=int, default=None,
                        help="override max_evidence_per_session (chunks per "
                             "session); the entry window in session terms is "
                             "top_k / cap, so this moves recall@k in the entry "
                             "view without touching retrieval")
    parser.add_argument("--candidate-per-session", type=int, default=None,
                        help="keep at most N candidates per session before "
                             "truncating the pool (0 = entry-major, the "
                             "default)")
    parser.add_argument("--recall-channel-depth", type=int, default=None,
                        help="entries pulled per recall channel when the pool "
                             "is session-major")
    parser.add_argument("--rerank-session-level", action="store_true",
                        help="score one representative document per session "
                             "instead of one per entry")
    parser.add_argument("--rerank-weight", type=float, default=None,
                        help="blend weight of the cross-encoder score")
    parser.add_argument("--rerank-model", type=str, default=None,
                        help="cross-encoder used by the reranking stage")
    parser.add_argument("--rerank-probability-scores", action="store_true",
                        help="the cross-encoder already emits 0..1 relevance, "
                             "so skip the temperature sigmoid")
    parser.add_argument("--rerank-span-tokens", type=int, default=None,
                        help="select the cross-encoder document by query-term "
                             "density instead of taking a character prefix")
    parser.add_argument("--rerank-top-n", type=int, default=None,
                        help="how many fused candidates go through the "
                             "cross-encoder (the latency knob)")
    parser.add_argument("--rerank-max-length", type=int, default=None,
                        help="pair cap the cross-encoder reads; a property of "
                             "the checkpoint (MiniLM 512, bge-reranker-v2-m3 "
                             "8194), not a tuning choice")
    parser.add_argument("--rerank-doc-tokens", type=int, default=None,
                        help="per-document token budget handed to the "
                             "cross-encoder, clipped with its own tokenizer")
    parser.add_argument("--dense-eligible", action="store_true",
                        help="score a candidate that ONLY the dense channel "
                             "found. Off by default, where dense can re-order "
                             "what lexical/entity found but never recall "
                             "anything of its own")
    parser.add_argument("--dense-eligible-sim", type=float, default=None,
                        help="absolute cosine floor for --dense-eligible "
                             "(default 0.45); implies --dense-eligible")
    parser.add_argument("--dense-eligible-max", type=int, default=None,
                        help="cap on dense-only candidates admitted per query "
                             "(0 = unlimited)")
    parser.add_argument("--dense-fill", action="store_true",
                        help="APPEND a few memories only dense reached, after "
                             "assembly. Unlike --dense-eligible this never "
                             "re-ranks or displaces anything the lexical "
                             "channels produced")
    parser.add_argument("--dense-fill-sim", type=float, default=None,
                        help="absolute cosine floor for --dense-fill "
                             "(default 0.50); implies --dense-fill")
    parser.add_argument("--dense-fill-max", type=int, default=None,
                        help="how many entries --dense-fill may append")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="override evidence_max_sessions (distinct sessions "
                             "in the payload). Needed to tell apart 'the fill "
                             "found new evidence' from 'the fill merely widened "
                             "a session cap that was binding'")
    parser.add_argument("--operative-weight", type=float, default=None,
                        help="weight of the fifth ranking term, 'this chunk "
                             "records an action' (0 = off, the default). Targets "
                             "the measured failure where a read of a file "
                             "outscores the edit that changed it")
    parser.add_argument("--position-weight", type=float, default=None,
                        help="override evidence_position_weight: intra-session "
                             "tilt of slot choice toward the end of the "
                             "trajectory. Cannot change which sessions are "
                             "returned, so the session view is a no-op check")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(
            f"error: {args.data} not found; run eval/build_benchmark.py first",
            file=sys.stderr,
        )
        return 2

    overrides: dict = {}
    if args.noise_gate is not None:
        overrides["min_evidence_score"] = args.noise_gate
    if args.budget_tokens is not None:
        overrides["evidence_budget_tokens"] = args.budget_tokens
    if args.full_count is not None:
        overrides["evidence_full_count"] = args.full_count
    if args.cap is not None:
        overrides["max_evidence_per_session"] = args.cap
    if args.item_tokens is not None:
        overrides["evidence_item_tokens"] = args.item_tokens
    if args.candidate_per_session is not None:
        overrides["candidate_per_session"] = args.candidate_per_session
    if args.recall_channel_depth is not None:
        overrides["recall_channel_depth"] = args.recall_channel_depth
    if args.rerank_session_level:
        overrides["rerank_session_level"] = True
    if args.rerank_weight is not None:
        overrides["rerank_weight"] = args.rerank_weight
    if args.rerank_model is not None:
        overrides["rerank_model"] = args.rerank_model
    if args.rerank_probability_scores:
        overrides["rerank_probability_scores"] = True
    if args.rerank_span_tokens is not None:
        overrides["rerank_span_tokens"] = args.rerank_span_tokens
    if args.rerank_top_n is not None:
        overrides["rerank_top_n"] = args.rerank_top_n
    if args.rerank_max_length is not None:
        overrides["rerank_max_length"] = args.rerank_max_length
    if args.rerank_doc_tokens is not None:
        overrides["rerank_doc_tokens"] = args.rerank_doc_tokens
    if args.dense_eligible:
        overrides["dense_eligible"] = True
    if args.dense_eligible_sim is not None:
        # Passing a floor without the flag would silently do nothing.
        overrides["dense_eligible"] = True
        overrides["dense_eligible_min_similarity"] = args.dense_eligible_sim
    if args.dense_eligible_max is not None:
        overrides["dense_eligible_max"] = args.dense_eligible_max
    if args.dense_fill:
        overrides["dense_fill"] = True
    if args.dense_fill_sim is not None:
        overrides["dense_fill"] = True
        overrides["dense_fill_min_similarity"] = args.dense_fill_sim
    if args.dense_fill_max is not None:
        overrides["dense_fill"] = True
        overrides["dense_fill_max"] = args.dense_fill_max
    if args.max_sessions is not None:
        overrides["evidence_max_sessions"] = args.max_sessions
    if args.operative_weight is not None:
        overrides["operative_rank_weight"] = args.operative_weight
    if args.position_weight is not None:
        overrides["evidence_position_weight"] = args.position_weight

    ks = tuple(int(k) for k in args.ks.split(",") if k.strip())
    data = load_benchmark(args.data)
    out = run(
        data,
        top_k=args.top_k,
        limit=args.limit,
        ks=ks,
        quiet=args.quiet,
        overrides=overrides or None,
    )
    if overrides:
        print(f"overrides: {overrides}")
    report(out, ks=ks, meta=data.get("meta", {}))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(
                {"meta": data.get("meta"), "metrics": out["metrics"],
                 "random_baseline": out["random_baseline"], "counts": out["counts"]},
                handle,
                indent=2,
            )
        print(f"\nwrote {args.out}")

    if args.dump_per_query:
        args.dump_per_query.parent.mkdir(parents=True, exist_ok=True)
        # Relevance is "touched at least one shared file", which makes a pair
        # that shares three files and one that shares a single hot file equally
        # "relevant" in the metrics. Dumping the overlap lets a low recall@k be
        # split into "weak labels we cannot reasonably rank" and "strong matches
        # we actually mis-ranked" — without it, recall@k is unactionable.
        session_files = {
            m["session_id"]: set(m.get("files") or ()) for m in data["memories"]
        }
        detail = []
        for query in data["queries"]:
            if query["query_id"] not in out["results"]:
                continue
            result = out["results"][query["query_id"]]
            query_files = set(query.get("files") or ())
            overlap = {
                session: len(query_files & session_files.get(session, set()))
                for session in result["relevant"]
            }
            detail.append(
                {
                    "query_id": query["query_id"],
                    "repo": query["repo"],
                    "n_relevant": len(result["relevant"]),
                    "n_returned": len(result["ranked"]),
                    "first_relevant_rank": next(
                        (
                            i
                            for i, s in enumerate(result["ranked"], start=1)
                            if s in result["relevant"]
                        ),
                        None,
                    ),
                    "ranked": result["ranked"],
                    "ranked_items": result["ranked_items"],
                    "relevant_overlap": overlap,
                    "query": query["query"][:300],
                }
            )
        with args.dump_per_query.open("w", encoding="utf-8") as handle:
            json.dump(detail, handle, indent=2, ensure_ascii=False)
        print(f"wrote {args.dump_per_query}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
