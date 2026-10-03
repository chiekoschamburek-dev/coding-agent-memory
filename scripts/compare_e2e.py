"""Paired comparison of two end-to-end Answer runs (``run_endtoend.py`` output).

Reads the ``outcomes`` lists from two run files, pairs them per ``query_id``,
and reports: majority-vote accuracy per arm, the exact McNemar test on
discordant pairs, the per-pass accuracy spread (the answer model's noise
floor), and the retrieval diagnostics that explain a gap when one shows up
(``n_shown`` context size, relevant sessions in context, answer session
present).

    python scripts/compare_e2e.py eval/results/e2e_codemem.json eval/results/e2e_rag.json
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(0, k + 1)) / 2**n * 2)


def load_outcomes(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    (label, condition), = data["results"].items()
    # Full per-question outcomes live under the file's top-level "outcomes"
    # key (the condition summary omits them); fall back to the condition for
    # older files.
    outcomes = data.get("outcomes", {}).get(label) or condition.get("outcomes")
    return {o["query_id"]: o for o in outcomes}, label


def main(argv: list[str]) -> int:
    base_path, other_path = Path(argv[0]), Path(argv[1])
    base, base_label = load_outcomes(base_path)
    other, other_label = load_outcomes(other_path)
    common = sorted(set(base) & set(other))
    if len(common) < len(base) or len(common) < len(other):
        print(f"note: pairing over the intersection, {len(common)} questions")

    b_acc = statistics.fmean(base[q]["correct"] for q in common)
    o_acc = statistics.fmean(other[q]["correct"] for q in common)
    b01 = sum(1 for q in common if not base[q]["correct"] and other[q]["correct"])
    b10 = sum(1 for q in common if base[q]["correct"] and not other[q]["correct"])
    p = mcnemar_exact(b10, b01)

    print(f"paired end-to-end comparison, n={len(common)} questions")
    print(f"  {base_label}: majority accuracy {b_acc:.3f}")
    print(f"  {other_label}: majority accuracy {o_acc:.3f}")
    print(f"  discordant: {other_label} +{b01} / -{b10}  exact McNemar p={p:.4f}")

    for name, arm in ((base_label, base), (other_label, other)):
        per_pass = arm[common[0]]["votes_correct"]
        n_pass = len(per_pass)
        spreads = [
            statistics.fmean(arm[q]["votes_correct"][i] for q in common)
            for i in range(n_pass)
        ]
        print(
            f"  {name}: per-pass accuracy "
            f"{min(spreads):.3f}..{max(spreads):.3f}, "
            f"unanimous {statistics.fmean(arm[q]['unanimous'] for q in common):.1%}"
        )

    # Bootstrap the difference over questions for a CI on the paired delta.
    rng = random.Random(20261002)
    diffs = []
    for _ in range(2000):
        idx = [rng.randrange(len(common)) for _ in range(len(common))]
        diffs.append(
            statistics.fmean(other[common[i]]["correct"] for i in idx)
            - statistics.fmean(base[common[i]]["correct"] for i in idx)
        )
    diffs.sort()
    print(
        f"  delta ({other_label} - {base_label}) "
        f"{o_acc - b_acc:+.3f}  95% CI [{diffs[49]:+.3f}, {diffs[1949]:+.3f}]"
    )

    print("\nretrieval diagnostics (means over questions):")
    for name, arm in ((base_label, base), (other_label, other)):
        print(
            f"  {name}: n_shown {statistics.fmean(arm[q]['n_shown'] for q in common):.1f}, "
            f"relevant in context {statistics.fmean(arm[q]['n_relevant_shown'] for q in common):.2f}, "
            f"answer session shown {statistics.fmean(arm[q]['answer_session_shown'] for q in common):.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
