"""Paired significance between two ``--dump-per-query`` runs.

The project retires variants at p ~ 0.25 rather than shipping them, so a raw
delta between two benchmark runs is not a result. This reads the per-query dumps
``eval/run_benchmark.py`` writes, recomputes each metric per query under both
configurations, and reports the paired bootstrap over the same 89 queries:
delta, 95% CI, two-sided p, and how many queries moved.

Both dumps must cover the same queries; only the intersection is paired, and the
tool says so when the intersection is smaller than either side.

    python scripts/paired_compare.py \\
        eval/results/per_query_off.json eval/results/per_query_on.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "eval")
sys.path.insert(0, "scripts")

from exp_entry_order import paired_bootstrap  # noqa: E402
from metrics import (  # noqa: E402
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def per_query(rows: list[dict], k: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        relevant = set(row["relevant_overlap"])
        if not relevant:
            continue
        entry = {
            "session recall@10": recall_at_k(row["ranked"], relevant, 10),
            "session ndcg@10": ndcg_at_k(row["ranked"], relevant, 10),
            "session mrr": reciprocal_rank(row["ranked"], relevant),
            "entry recall@10": recall_at_k(row["ranked_items"], relevant, k),
            "entry ndcg@10": ndcg_at_k(row["ranked_items"], relevant, k),
            "entry precision@10": precision_at_k(row["ranked_items"], relevant, k),
            "entry mrr": reciprocal_rank(row["ranked_items"], relevant),
            "payload entries": float(len(row["ranked_items"])),
            "payload sessions": float(len(row["ranked"])),
        }
        strong = {
            s for s, overlap in row["relevant_overlap"].items() if overlap >= 2
        }
        if strong:
            entry["STRONG entry recall@10"] = recall_at_k(
                row["ranked_items"], strong, k
            )
            entry["STRONG session recall@10"] = recall_at_k(row["ranked"], strong, 10)
        out[row["query_id"]] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("-k", type=int, default=10, help="entry window (default 10)")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    a = per_query(json.loads(args.before.read_text(encoding="utf-8")), args.k)
    b = per_query(json.loads(args.after.read_text(encoding="utf-8")), args.k)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("no query ids in common; dumps are from different corpora", file=sys.stderr)
        return 1
    if len(shared) < min(len(a), len(b)):
        print(
            f"warning: pairing {len(shared)} of {len(a)}/{len(b)} queries",
            file=sys.stderr,
        )

    # STRONG-* keys exist only for queries that have a strong-relevant session, so
    # each metric is paired over the queries carrying it rather than over a
    # smallest-common set that would drop those rows entirely.
    metrics = sorted({key for q in shared for key in a[q]})
    print()
    print(
        f"{'metric':26s}{'before':>9}{'after':>9}{'delta':>10}"
        f"{'95% CI':>20}{'p':>8}{'moved':>7}{'n':>5}"
    )
    print("-" * 95)
    for metric in metrics:
        keys = [q for q in shared if metric in a[q] and metric in b[q]]
        before = {q: a[q][metric] for q in keys}
        after = {q: b[q][metric] for q in keys}
        stats = paired_bootstrap(
            before, after, resamples=args.resamples, seed=args.seed
        )
        ci = f"[{stats['ci_low']:+.4f},{stats['ci_high']:+.4f}]"
        print(
            f"{metric:26s}"
            f"{sum(before.values()) / len(before):>9.4f}"
            f"{sum(after.values()) / len(after):>9.4f}"
            f"{stats['delta']:>+10.4f}"
            f"{ci:>20}{stats['p']:>8.3f}{stats['queries_moved']:>7d}{stats['n']:>5d}"
        )
    print()
    print(f"paired over {len(shared)} queries, {args.resamples} resamples, seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
