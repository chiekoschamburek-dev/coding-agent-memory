"""Split the entry-unit recall loss into retrieval / ranking / packing.

Reads a ``--dump-per-query`` JSON produced by ``eval/run_benchmark.py`` and
answers one question: when a relevant session is not inside the top-k entry
window, why not?

  * it never entered the payload at all          -> retrieval miss
  * it entered, but sits below the window        -> recoverable by reordering
  * it entered, but the window has no slot left  -> packing (per-session cap)
  * the window is simply smaller than the answer -> nothing to recover

The decomposition respects the real slot constraint: a session spends up to
``--cap`` of the k slots, so a window cannot hold ``min(avail, k)`` distinct
sessions.  The ceiling is computed greedily (shortest sessions first).

The cap replay rebuilds the entry sequence at a different per-session cap.
This is exact rather than approximate: the assembler emits sessions in rank
order and takes up to ``cap`` chunks of each, and the session set is decided by
the noise gate, not the cap -- so lowering the cap only shortens each session's
run, it does not change which sessions are present.  Verified against the real
run at ``--cap 3``.

    python scripts/diagnose_entry_recall.py eval/results/per_query_p2c.json
    python scripts/diagnose_entry_recall.py eval/results/per_query_p2c.json -k 20
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics


def macro(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def ranked_items_at_cap(query: dict, cap: int) -> list[str]:
    """Rebuild the entry sequence as if ``max_evidence_per_session`` were cap."""
    counts = collections.Counter(query["ranked_items"])
    return [s for s in query["ranked"] for _ in range(min(cap, counts[s]))]


def decompose(queries: list[dict], k: int, cap: int) -> dict[str, list[float]]:
    """Per-query shares of the relevant sessions, under the cap's slot budget."""
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for q in queries:
        rel = set(q["relevant_overlap"])
        n = len(rel)
        items = ranked_items_at_cap(q, cap) if cap != 3 else q["ranked_items"]
        counts = collections.Counter(items)
        actual = len(set(items[:k]) & rel)

        # Distinct relevant sessions that fit: shortest sessions first, greedy.
        avail = sorted(counts[s] for s in rel if s in counts)
        used = fit = 0
        for c in avail:
            if used + c <= k:
                used += c
                fit += 1

        acc["actual"].append(actual / n)
        acc["fit"].append(fit / n)
        acc["ranking"].append((fit - actual) / n)
        acc["packing"].append((len(avail) - fit) / n)
        acc["retrieval"].append((n - len(avail)) / n)
        acc["span"].append(len(set(items[:k])))
        acc["slots_now"].append(sum(1 for s in items[:k] if s in rel) / min(k, len(items)))
        chunks = sum(counts[s] for s in rel if s in counts)
        acc["slots_ideal"].append(min(k, chunks) / k)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="per-query JSON from --dump-per-query")
    ap.add_argument("-k", type=int, default=10, help="entry window (default 10)")
    args = ap.parse_args()

    k = args.k
    with open(args.dump, encoding="utf-8") as fh:
        rows = json.load(fh)
    queries = [q for q in rows if q["relevant_overlap"]]

    # ---------------------------------------------------------------- sanity
    # The dump tags every entry slot with its source session id, so a relevant
    # session can be located in the window without re-running the search.
    print(f"dump                : {args.dump}")
    print(f"queries             : {len(rows)} ({len(queries)} with a relevant session)")
    print(f"session list unique : {all(len(q['ranked']) == len(set(q['ranked'])) for q in rows)}")
    print(f"entry slots resolve : {all(set(q['ranked_items']) <= set(q['ranked']) for q in rows)}")
    print(f"window k            : {k} entries")
    print(f"payload ends at     : {macro([len(q['ranked_items']) for q in queries]):.1f} entries / "
          f"{macro([len(q['ranked']) for q in queries]):.1f} sessions "
          f"(the noise gate ends it, not top_k)")

    # --------------------------------------------- decomposition at the cap 3
    acc = decompose(queries, k, cap=3)
    n_rel = sum(len(q["relevant_overlap"]) for q in queries)
    tot = {key: sum(len(q["relevant_overlap"]) * v for q, v in zip(queries, vals))
           for key, vals in acc.items() if key not in ("span", "slots_now", "slots_ideal")}

    section(f"1. item recall@{k} decomposed under the real slot budget (cap=3)")
    print(f"  {'bucket':44s}{'macro':>9}{'micro':>9}")
    for key, label in [("actual", "in window  (achieved)"),
                       ("ranking", "in payload, recoverable by reordering"),
                       ("packing", "in payload, no slot left at cap 3"),
                       ("retrieval", "never entered the payload")]:
        print(f"  {label:44s}{macro(acc[key]):>9.1%}{tot[key] / n_rel:>9.1%}")
    print(f"  {'ceiling (perfect reorder, same payload)':44s}"
          f"{macro(acc['fit']):>9.1%}{tot['fit'] / n_rel:>9.1%}")
    print()
    short = 1 - macro(acc["actual"])
    print(f"  share of the {short * 100:.0f}-point shortfall each bucket owns (macro):")
    for key, label in [("ranking", "reordering"), ("packing", "packing (cap)"),
                       ("retrieval", "retrieval")]:
        print(f"    {label:14s}{macro(acc[key]) / short:>7.1%}  ({macro(acc[key]) * 100:>4.1f} pts)")
    print()
    print(f"  the window holds {macro(acc['slots_now']) * k:.2f} relevant-entry slots now; "
          f"a perfect reorder of the same payload reaches {macro(acc['slots_ideal']) * k:.2f}")
    print(f"  -> ordering reclaims {macro(acc['slots_ideal']) - macro(acc['slots_now']):.1%} of the window, "
          f"the last {1 - macro(acc['slots_ideal']):.1%} is absence, not crowding")

    # ----------------------------------------------------------- cap replay
    section("2. the packing lever: what the per-session cap costs")
    print(f"  {'cap':>4}{'item recall@k':>16}{'sessions spanned':>19}{'equals':>22}")
    for cap in (1, 2, 3):
        a = decompose(queries, k, cap=cap)
        got = macro(a["actual"])
        span = macro(a["span"])
        sess = macro([len(set(q["ranked"][:max(1, round(span))]) & set(q["relevant_overlap"]))
                      / len(q["relevant_overlap"]) for q in queries])
        print(f"  {cap:>4}{got:>16.4f}{span:>19.2f}{f'session recall@{round(span)} = {sess:.4f}':>22}")
    print("  (the cap 3 row reads the dump verbatim; the other rows are replayed above it)")
    print()
    print(f"  {'cap':>4}{'achieved':>10}{'reorder':>10}{'packing':>10}{'retrieval':>11}{'ceiling':>10}")
    for cap in (1, 2, 3):
        a = decompose(queries, k, cap=cap)
        print(f"  {cap:>4}{macro(a['actual']):>10.3f}"
              f"{macro(a['ranking']):>10.3f}{macro(a['packing']):>10.3f}"
              f"{macro(a['retrieval']):>11.3f}{macro(a['fit']):>10.3f}")
    print()
    print("  the session set is gate-decided, so lowering the cap costs no retrieval:")
    print("  it only widens the window in session terms. Compare the end-to-end answer")
    print("  score before changing the default -- the session view barely moves.")

    # ------------------------------------------------------- by query size
    section(f"3. recall@{k} by number of relevant sessions")
    by_size: dict[int, list[float]] = collections.defaultdict(list)
    for q in queries:
        by_size[min(len(q["relevant_overlap"]), 4)].append(
            len(set(q["ranked_items"][:k]) & set(q["relevant_overlap"])) / len(q["relevant_overlap"])
        )
    print(f"  {'n_rel':>7}{'queries':>9}{'recall':>10}{'span':>8}")
    for size, vals in sorted(by_size.items()):
        label = str(size) if size < 4 else "4+"
        subset = [q for q in queries if min(len(q["relevant_overlap"]), 4) == size]
        span = macro([len(set(q["ranked_items"][:k])) for q in subset])
        print(f"  {label:>7}{len(vals):>9}{macro(vals):>10.1%}{span:>8.2f}")


if __name__ == "__main__":
    main()
