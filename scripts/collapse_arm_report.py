"""Read the conflict-collapse arm against its pre-registered decision rule.

The rule (drop every payload item whose text matches an option, when two or more
distinct options are matched) was calibrated before it was built: at tau=0.97 the
serve-time matcher agrees with the verbatim label on the branch quantity for
70/70 queries and fires on exactly the 35 conflicted ones. That calibration dump
is what this script re-reads, so the fired set is the *pre-registered* one and
not something chosen after seeing the arm.

The arm was pre-registered with bands, not a point prediction, because the
residual payload after the drop is unmeasured:

  very pessimistic  the remainder re-leads to the rival  ...... -1.3 pp
  conservative      the remainder is neutral             ...... +2.1 pp
  moderate/opt.     the remainder behaves like purity-1 . ...... +12.3/+14.3 pp

Pass = gains > losses on the 70 (exact McNemar) AND the answer-absent half of the
fired set actually improves. Fail = the whole set does not beat the recorded
baseline AND that half is unmoved; that would mean the collapsed payload is
*not* neutral, and it kills the last serve-time lever rather than this arm.

The not-fired half is the attribution control. Collapse cannot touch a payload
that matches fewer than two options, so anything other than answer-model noise
moving those questions is a bug in the gate, not an effect.

    PYTHONPATH=src python scripts/collapse_arm_report.py \\
        eval/results/e2eCT_ms2.json eval/results/e2eCT_collapse.json \\
        --match eval/results/option_match.json --tau 0.97
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def load_arm(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    (label, condition), = data["results"].items()
    outcomes = data.get("outcomes", {}).get(label) or condition.get("outcomes")
    return {o["query_id"]: o for o in outcomes}


def fired_at(match: Path, tau: float) -> tuple[set[str], dict[str, int]]:
    """Queries whose served payload matches two or more options at `tau`.

    Every scored query is a key of the returned map, including the ones that
    match nothing - the control group needs them, and a payload matching zero
    options is the definition of a clean payload, not a missing observation.
    """
    data = json.loads(match.read_text(encoding="utf-8"))
    per: dict[str, set[int]] = {s["query_id"]: set() for s in data["states"]}
    for p in data["pairs"]:
        if p["cos"] >= tau:
            per.setdefault(p["query_id"], set()).add(p["option"])
    return (
        {qid for qid, opts in per.items() if len(opts) >= 2},
        {qid: len(opts) for qid, opts in per.items()},
    )


def report(name: str, ids: list[str], base: dict, arm: dict) -> None:
    if not ids:
        print(f"  {name:<26} n=0")
        return
    b = statistics.fmean(base[q]["correct"] for q in ids)
    a = statistics.fmean(arm[q]["correct"] for q in ids)
    gains = [q for q in ids if not base[q]["correct"] and arm[q]["correct"]]
    losses = [q for q in ids if base[q]["correct"] and not arm[q]["correct"]]
    p = mcnemar_exact(len(losses), len(gains))
    print(
        f"  {name:<26} n={len(ids):>2}  base {b:.3f} -> arm {a:.3f} "
        f"({a - b:+.3f})  +{len(gains)}/-{len(losses)}  p={p:.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("arm", type=Path)
    ap.add_argument("--match", type=Path, default=Path("eval/results/option_match.json"))
    ap.add_argument("--tau", type=float, default=0.97)
    args = ap.parse_args()

    base, arm = load_arm(args.base), load_arm(args.arm)
    common = sorted(set(base) & set(arm))
    fired, n_matched = fired_at(args.match, args.tau)
    scored = [q for q in common if q in n_matched]

    print(f"paired over {len(common)} questions; fired set at tau={args.tau}: "
          f"{len([q for q in scored if q in fired])} of {len(scored)}")

    report("whole set", common, base, arm)
    report("fired (conflict)", [q for q in scored if q in fired], base, arm)
    not_fired = [q for q in scored if q not in fired]
    report("not fired (control)", not_fired, base, arm)

    firing = [q for q in scored if q in fired]
    absent = [q for q in firing if not base[q]["answer_session_shown"]]
    present = [q for q in firing if base[q]["answer_session_shown"]]
    report("  fired, answer absent", absent, base, arm)
    report("  fired, answer present", present, base, arm)

    print("\npayload effect (mean items served):")
    for name, ids in (("fired", firing), ("not fired", not_fired)):
        if not ids:
            continue
        b = statistics.fmean(base[q]["n_shown"] for q in ids)
        a = statistics.fmean(arm[q]["n_shown"] for q in ids)
        print(f"  {name:<10} {b:.2f} -> {a:.2f}  ({a - b:+.2f})")
    print("  (not-fired must be ~0; that is the gate, not the answer model)")

    rng = random.Random(20261008)
    diffs = []
    for _ in range(2000):
        idx = [rng.randrange(len(common)) for _ in range(len(common))]
        diffs.append(
            statistics.fmean(arm[common[i]]["correct"] for i in idx)
            - statistics.fmean(base[common[i]]["correct"] for i in idx)
        )
    diffs.sort()
    delta = statistics.fmean(arm[q]["correct"] for q in common) - statistics.fmean(
        base[q]["correct"] for q in common
    )
    print(f"\n  whole-set delta {delta:+.3f}  95% CI "
          f"[{diffs[49]:+.3f}, {diffs[1949]:+.3f}]")

    gains = sum(1 for q in common if not base[q]["correct"] and arm[q]["correct"])
    losses = sum(1 for q in common if base[q]["correct"] and not arm[q]["correct"])
    abs_gain = sum(1 for q in absent if not base[q]["correct"] and arm[q]["correct"])
    abs_loss = sum(1 for q in absent if base[q]["correct"] and not arm[q]["correct"])
    print("\ndecision rule:")
    print(f"  (a) gains > losses on the 70 .......... {'PASS' if gains > losses else 'FAIL'}"
          f"  ({gains} vs {losses})")
    print(f"  (b) answer-absent half improves ....... {'PASS' if abs_gain > abs_loss else 'FAIL'}"
          f"  (+{abs_gain}/-{abs_loss} on {len(absent)} questions)")
    if delta <= 0 and abs_gain <= abs_loss:
        print("  => both fail: the collapsed payload is NOT neutral. The rule closes,")
        print("     and with it the last serve-time-observable lever on this anchor.")
    elif gains > losses and abs_gain > abs_loss:
        print("  => both pass: the collapse is a real, bounded gain. Price it on the")
        print("     sealed anchor before it ships. A McNemar p above 0.05 on n=70")
        print("     still means 'directional, same sign as the mechanism'.")
    else:
        print("  => mixed: one condition met, one not. Do not ship; record which half")
        print("     of the mechanism failed to show.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
