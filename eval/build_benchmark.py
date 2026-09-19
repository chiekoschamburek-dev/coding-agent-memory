"""Local proxy benchmark.

CAMBench Coding — the scored suite — is not public, so we cannot optimize
against it directly. This harness builds a *proxy* corpus in which the two
properties the coding track is described as testing are reproduced:

* **same-repository distractors** — every distractor is drawn from the same
  repository as the relevant memory, so it shares vocabulary, file paths, and
  style;
* **four noise levels** — low / medium-low / medium-high / high, matching the
  competition description.

Relevance is defined by construction, not by a model, so the metrics are stable
and the conclusions are about retrieval rather than about judging.

Status: the builder is intentionally unfinished — it is the W2 deliverable. The
not-yet-implemented pieces are marked TODO and raise rather than silently
returning empty results, so a partial run cannot be mistaken for a real one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

# TODO(W2): implement the corpus builder. The intended design:
#
#   source   public SWE-bench Lite / Verified instances (MIT). Used internally
#            for development only: excluded from the image, never redistributed,
#            never used for training or dataset reconstruction.
#
#   relevant for each task, the trajectory derived from that task's own
#            resolution (problem statement + the diff that resolved it)
#
#   distractors  trajectories from *other tasks in the same repository*,
#            which is what makes the noise realistic — a distractor from a
#            different repository is trivially separable and tests nothing
#
#   levels   low 0.2x / medium-low 0.5x / medium-high 1x / high 2x distractors
#            per relevant memory
#
#   output   {"queries": [{"query_id", "query", "options"?, "relevant": [ids]}],
#             "memories": [{"id", "user_id", "session_id", "messages": [...]}]}
#
# The files are then replayed through the live Add/Search API and scored with
# eval/metrics.py. Ablations disable one component at a time (identifier
# channel, IDF weighting, noise gate, intent bonus) to attribute any gain.

NOISE_LEVELS = {
    "low": 0.2,
    "medium-low": 0.5,
    "medium-high": 1.0,
    "high": 2.0,
}


@dataclass
class BenchmarkCase:
    query_id: str
    query: str
    relevant: list[str] = field(default_factory=list)
    options: list[str] | None = None


@dataclass
class BenchmarkMemory:
    id: str
    user_id: str
    session_id: str
    messages: list[dict]
    relevant_to: list[str] = field(default_factory=list)


def build(source: Path, out: Path, *, levels: list[str] | None = None) -> None:
    """Build the proxy benchmark from a public dataset checkout."""
    raise NotImplementedError(
        "The proxy benchmark builder is the W2 deliverable. "
        "See the module docstring for the intended design; until it is "
        "implemented, metrics cannot be computed for a local run."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="dataset checkout")
    parser.add_argument("--out", type=Path, required=True, help="output json")
    parser.add_argument(
        "--levels", default=",".join(NOISE_LEVELS), help="comma-separated noise levels"
    )
    args = parser.parse_args(argv)

    levels = [item.strip() for item in args.levels.split(",") if item.strip()]
    unknown = [level for level in levels if level not in NOISE_LEVELS]
    if unknown:
        parser.error(f"unknown noise levels: {unknown}; choose from {list(NOISE_LEVELS)}")

    try:
        build(args.source, args.out, levels=levels)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
