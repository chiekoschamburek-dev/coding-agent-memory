"""The second scored condition: needle-only history.

AML's Coding Memory track runs **150 tasks under both relevant-history and
noisy-history settings — 300 task-condition units** (Cycle 2 announcement). Every
measurement in this tree so far covers the noisy side only, so half the graded
surface is unmeasured.

This builds the missing condition honestly and cheaply. The corpus is trimmed to
the sessions that the file-overlap relevance label calls relevant to *some*
scored query — 149 of 300 sessions — and everything else is held identical: same
queries, same ranking code, same scoring, same assembly. What changes is only
what the haystack is made of. Retrieval still has to discriminate: a query's own
sessions are needles among 149 needles, not a single candidate.

Read it as a decomposition, not as an official number:

* **relevant condition** — can the system find and deliver the right history when
  the history it can reach is all usable? This is the practical upper bound of
  the retrieval+assembly half, and it isolates utilisation from search: if the
  answer session is retrieved and the payload is still not decidable, the failure
  is downstream of retrieval.
* **the gap between the two** — what the distractors cost. That difference is a
  scored quantity in the real suite, which is why it is worth having a local
  estimate of it rather than only the noisy number.

Usage::

    PYTHONPATH=src python eval/build_relevant_condition.py
    PYTHONPATH=src python eval/run_benchmark.py \\
        --data eval/data/benchmark_relevant.json --limit 20 \\
        --dump-per-query eval/results/pq_relevant.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = ROOT / "eval" / "data" / "benchmark.json"
DEFAULT_OUT = ROOT / "eval" / "data" / "benchmark_relevant.json"


def build(source: Path, *, min_per_repo: int = 1) -> tuple[dict, dict]:
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)

    scored = [q for q in data["queries"] if q.get("relevant")]
    needles = {session for query in scored for session in query["relevant"]}

    by_session = {m["session_id"]: m for m in data["memories"]}

    # The relevance label is defined per repository, so a needle must already be
    # in the same repository as the queries that name it. Assert rather than
    # assume: a cross-repo needle would make the trimmed corpus incoherent under
    # the user_id = bench:<repo> isolation this tree relies on.
    repo_of = {m["session_id"]: m["repo"] for m in data["memories"]}
    mismatched = 0
    for query in scored:
        for session in query["relevant"]:
            if repo_of.get(session) != query["repo"]:
                mismatched += 1
    if mismatched:
        raise SystemExit(
            f"{mismatched} relevant sessions are not in their query's repository; "
            "the label or the corpus changed and this trimmer is no longer sound"
        )

    kept = [by_session[s] for s in sorted(needles) if s in by_session]
    per_repo = Counter(m["repo"] for m in kept)
    thin = {repo: n for repo, n in per_repo.items() if n < min_per_repo}

    meta = dict(data["meta"])
    meta["condition"] = "relevant-history (needle-only corpus)"
    meta["derived_from"] = source.name
    meta["derived_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["trimming"] = (
        "Kept every session the file-overlap label calls relevant to at least one "
        "scored query, and dropped the rest. Queries, relevance labels and all "
        "ranking code are unchanged, so the two conditions differ only in what "
        "the corpus contains."
    )
    meta["not_the_scored_benchmark"] = (
        "AML's own relevant-history condition is not public in its construction "
        "details; this is OUR trim, used to decompose local scores. It must not be "
        "presented as an official number."
    )
    meta["counts"] = {
        "memories": len(kept),
        "memories_dropped": len(data["memories"]) - len(kept),
        "queries_scored": len(scored),
        "needles": len(needles),
    }

    payload = {"meta": meta, "memories": kept, "queries": data["queries"]}
    report = {
        "kept": len(kept),
        "dropped": len(data["memories"]) - len(kept),
        "queries": len(scored),
        "per_repo": dict(sorted(per_repo.items())),
        "thin_repos": thin,
    }
    return payload, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"error: {args.source} not found", file=sys.stderr)
        return 2

    payload, report = build(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)

    print(f"kept {report['kept']} of {report['kept'] + report['dropped']} sessions "
          f"({report['kept'] / (report['kept'] + report['dropped']):.1%}), "
          f"{report['queries']} queries unchanged")
    print("needles per repository:")
    for repo, count in report["per_repo"].items():
        print(f"  {repo:<38}{count:>4}")
    if report["thin_repos"]:
        print(f"  note: {len(report['thin_repos'])} repositories are thin "
              f"(< {1} needle) and will return little")
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
