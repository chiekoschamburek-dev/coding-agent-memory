"""Phase A on the menu dump: shortlist diversification, offline.

The shipped menu is the score-greedy top-8 (``order[:8]``). The recorded
decomposition blames 20 of select-llm's 31 misses on the answer session not
being in that menu — same-file distractors crowd it out. Diversification
tests whether a coverage-greedy menu (skip a session whose file manifest
nearly duplicates one already seated) seats the answer session more often.

The judge-is-non-monotone lesson is enforced by the metric: every variant
reports GAINS and LOSSES against the shipped menu across all 58 questions,
never only the 20 misses — a rule that seats three new answer sessions by
unseating four currently-correct ones is a loss, whatever the gross gain.

Also settled on the same dump: the corrected-F3 verdict. The archived
rejection ("signal too sparse, exactly 0.4746") measured a regex that could
not match prose; with the repaired lexicon, does adding cause-x-identifier
to the fusion move the payload reach?

Usage::

    PYTHONPATH=src python scripts/menu_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MENU = 8


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def diversified(sessions: list[dict], theta: float) -> list[str]:
    """Coverage-greedy menu: seat in rank order, skip near-duplicate files.

    Skipped candidates backfill only if the pool runs out, so the menu is
    always 8 seats when 8 sessions exist.
    """
    seated: list[dict] = []
    skipped: list[dict] = []
    for s in sessions:
        if len(seated) >= MENU:
            break
        manifests = [set(t["files"]) for t in seated]
        if any(jaccard(set(s["files"]), m) >= theta for m in manifests):
            skipped.append(s)
        else:
            seated.append(s)
    for s in skipped:
        if len(seated) >= MENU:
            break
        seated.append(s)
    return [s["sid"] for s in seated[:MENU]]


def cap_per_cluster(sessions: list[dict], theta: float, cap: int) -> list[str]:
    """Variant: at most ``cap`` seats per overlapping-file cluster."""
    seated: list[dict] = []
    skipped: list[dict] = []
    for s in sessions:
        if len(seated) >= MENU:
            break
        similar = sum(
            1 for t in seated if jaccard(set(s["files"]), set(t["files"])) >= theta
        )
        (skipped if similar >= cap else seated).append(s)
    for s in skipped:
        if len(seated) >= MENU:
            break
        seated.append(s)
    return [s["sid"] for s in seated[:MENU]]


def main() -> int:
    dump = json.loads(
        (ROOT / "eval/results/menu_dump.json").read_text(encoding="utf-8")
    )
    rows = dump["rows"]

    missed = [r for r in rows if r["answer_session"] and not r["sel_shown"]]
    bottleneck = {
        r["query_id"] for r in missed
        if r["answer_session"] not in {s["sid"] for s in r["sessions"][:MENU]}
    }
    print(f"rows {len(rows)}; select-llm misses {len(missed)}; "
          f"shortlist-bottleneck {len(bottleneck)} (must match the archived 20)")

    def menu_of(r: dict) -> list[str]:
        return [s["sid"] for s in r["sessions"][:MENU]]

    def fused_menu(r: dict) -> list[str]:
        # the fused12 payload order restricted to the deep list is the fused
        # menu approximation; membership is what matters, not position
        deep = {s["sid"] for s in r["sessions"]}
        return [sid for sid in r["payloads"]["fused12"] if sid in deep][:MENU]

    variants: dict[str, object] = {}
    for theta in (0.2, 0.3, 0.4, 0.5):
        variants[f"jaccard>={theta}"] = (
            lambda r, t=theta: diversified(r["sessions"], t)
        )
    variants["cap2@0.3"] = lambda r: cap_per_cluster(r["sessions"], 0.3, 2)
    variants["fused12-menu"] = fused_menu

    report: dict[str, dict] = {}
    for name, fn in variants.items():
        gain, loss = [], []
        for r in rows:
            ans = r["answer_session"]
            if not ans:
                continue
            base_in = ans in menu_of(r)
            new_in = ans in fn(r)
            if new_in and not base_in:
                gain.append(r["query_id"])
            if base_in and not new_in:
                loss.append(r["query_id"])
        bottleneck_hit = sum(
            1 for r in rows
            if r["query_id"] in bottleneck and r["answer_session"] in fn(r)
        )
        report[name] = {
            "gain": len(gain),
            "loss": len(loss),
            "net": len(gain) - len(loss),
            "gain_ids": gain,
            "loss_ids": loss,
            "bottleneck_seated": bottleneck_hit,
        }
        print(f"{name:16s} gain {len(gain):2d}  loss {len(loss):2d}  "
              f"net {len(gain) - len(loss):+d}  "
              f"(of the {len(bottleneck)} bottleneck misses, "
              f"{bottleneck_hit} seated)")

    # ---- corrected-F3 verdict -------------------------------------------
    print("\ncorrected F3 (cause x identifier, repaired lexicon):")
    n_f3_sessions = sum(
        1 for r in rows for s in r["sessions"] if s["f3"] > 0
    )
    n_sessions = sum(len(r["sessions"]) for r in rows)
    print(f"  sessions with f3>0: {n_f3_sessions}/{n_sessions} "
          f"({100 * n_f3_sessions / n_sessions:.0f}%)")

    def reach(payload_key: str) -> tuple[int, list[str]]:
        hit, ids = 0, []
        for r in rows:
            if r["answer_session"] and r["answer_session"] in r["payloads"][payload_key]:
                hit += 1
                ids.append(r["query_id"])
        return hit, ids

    for key in ("shipped", "fused12", "fused123c", "f3alone"):
        hit, ids = reach(key)
        print(f"  payload reach {key:10s}: {hit}/{len(rows)}")

    _, s_ids = reach("shipped")
    _, f_ids = reach("fused12")
    _, c_ids = reach("fused123c")
    print(f"  fused123c vs fused12: +{len(set(c_ids) - set(f_ids))} "
          f"/-{len(set(f_ids) - set(c_ids))}")

    out = ROOT / "eval/results/menu_analysis.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump({
            "bottleneck_ids": sorted(bottleneck),
            "variants": report,
            "f3_sessions_nonzero": n_f3_sessions,
            "f3_sessions_total": n_sessions,
            "reach": {
                key: reach(key)[0] for key in
                ("shipped", "fused12", "fused123c", "f3alone")
            },
        }, handle, ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
