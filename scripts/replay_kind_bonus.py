"""Is the kind bonus pointed the wrong way?

Motivation (measured, `eval/results/entry_level.json`, 99 gold-vs-winner pairs over
38 queries): among every per-entry feature the pipeline already computes, the only
one where the gold chunk beats the chunk that outranked it is `kind == code`
(net +31.3 pp), and the strongest loser is `kind == diff` (net -24.2 pp). The gold
chunk is a `code` block 62.6% of the time and a `diff` 1.0% of the time; the chunk
that beat it is a `diff` 25.3% of the time. Meanwhile `INTENT_KIND_BONUS["debug"]`
credits diff/stacktrace/log/cmd and gives `code` nothing.

This script replays the ranking under alternative bonus tables, corpus built once,
and prices each arm against the shipped one with an exact paired sign test on the
two session-level outcomes the bonus is supposed to move:

  head_is_gold  - the chunk carrying a relevant session's score names a task file
                  (the shipped rate is 12.9%, `issue.md` section II)
  gold_in_window- some gold entry of a relevant session reaches the top-k window
  gold_emitted  - ... and survives into the shipped payload

Control arm: every bonus set to 1.0. If the shipped table is worth anything at all,
turning it off must cost something; if the arms are indistinguishable from the
control, the bonus is noise rather than a lever.

Run (dry, small corpus):
    PYTHONPATH=src python scripts/replay_kind_bonus.py --sessions 20 --limit 12
Run (full):
    PYTHONPATH=src python scripts/replay_kind_bonus.py --out eval/results/kind_bonus.json

Settings come from dataclass defaults, not `.env`, so this reads the shipped config
(cap 5 / 2 sessions / dense + rerank on) and not the cap-3 pin in the local .env.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

logging.disable(logging.WARNING)

from codemem.api.app import Container  # noqa: E402
from codemem.core.config import Settings  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402
import codemem.search.evidence as ev  # noqa: E402
from diagnose_entry_level import analyse, build, report  # noqa: E402

SHIPPED = {
    "debug": {"stacktrace": 1.18, "diff": 1.15, "test": 1.10, "log": 1.08, "cmd": 1.04},
    "develop": {"diff": 1.18, "code": 1.12, "prose": 1.06, "test": 1.05},
    "general": {},
}

ARMS: dict[str, dict[str, dict[str, float]]] = {
    "shipped": SHIPPED,
    # control: no kind bonus anywhere
    "bonus-off": {"debug": {}, "develop": {}, "general": {}},
    # surgical: under `debug`, promote `code` to what `diff` had and drop the
    # action-kind lifts. Leaves `develop` alone.
    "debug-code-up": {
        "debug": {"code": 1.15, "test": 1.10},
        "develop": SHIPPED["develop"],
        "general": {},
    },
    # maximal: same inversion applied to both intents
    "both-code-up": {
        "debug": {"code": 1.18, "prose": 1.06, "test": 1.10},
        "develop": {"code": 1.18, "prose": 1.06, "test": 1.05},
        "general": {},
    },
    # direction check: keep the shipped shape but make `diff` even stronger, so a
    # null here cannot be read as "the metric is blind to kind weights"
    "diff-harder": {
        "debug": {"stacktrace": 1.25, "diff": 1.30, "test": 1.10, "log": 1.15, "cmd": 1.08},
        "develop": SHIPPED["develop"],
        "general": {},
    },
}

OUTCOMES = ("head_is_gold", "gold_in_window", "gold_emitted", "gold_admitted")


def sign_test(discordant_plus: int, discordant_minus: int) -> float:
    """Two-sided exact binomial p for a paired binary comparison (ties dropped)."""
    n = discordant_plus + discordant_minus
    if n == 0:
        return 1.0
    k = min(discordant_plus, discordant_minus)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["query_id"], p["session_id"]): p for r in rows for p in r["pairs"]}


def detail_index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(r["query_id"], d["memory_id"]): d for r in rows for d in r["detail"]}


def shifts(base: dict, arm: dict) -> dict:
    """Instrument-sensitivity check: did the arm move ANY entry at all?

    The four session-level outcomes are coarse, and under `debug` the shipped table
    gives `code` and `prose` a 1.0 bonus, so `bonus-off` is byte-identical to
    `shipped` for candidates of those kinds. A null on the outcomes therefore means
    one of two different things -- the lever is dead, or no bonus-bearing kind sat
    where it mattered. This counts rank changes so the two can be told apart.
    """
    keys = set(base) & set(arm)
    moved = [(base[k]["rank"], arm[k]["rank"], base[k].get("kind")) for k in keys]
    changed = [m for m in moved if m[0] != m[1]]
    by_kind: dict[str, list[tuple[int, int]]] = {}
    for old, new, kind in changed:
        by_kind.setdefault(kind or "?", []).append((old, new))
    return {
        "entries_compared": len(keys),
        "entries_moved": len(changed),
        "mean_abs_rank_change_moved": (
            round(sum(abs(o - n) for o, n, _ in changed) / len(changed), 2) if changed else 0.0
        ),
        "moved_by_kind": {
            k: {
                "n": len(v),
                "improved": sum(1 for o, n in v if n < o),
                "worsened": sum(1 for o, n in v if n > o),
            }
            for k, v in sorted(by_kind.items())
        },
    }


def paired(a: dict, b: dict, field: str) -> dict:
    keys = set(a) & set(b)
    plus = sum(1 for k in keys if b[k].get(field) and not a[k].get(field))
    minus = sum(1 for k in keys if a[k].get(field) and not b[k].get(field))
    rate_a = sum(1 for k in keys if a[k].get(field)) / len(keys) if keys else 0.0
    rate_b = sum(1 for k in keys if b[k].get(field)) / len(keys) if keys else 0.0
    return {
        "field": field,
        "n_pairs": len(keys),
        "baseline_rate": round(rate_a, 4),
        "arm_rate": round(rate_b, 4),
        "gained": plus,
        "lost": minus,
        "p": round(sign_test(plus, minus), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="cap on queries")
    ap.add_argument("--sessions", type=int, default=None, help="cap on corpus sessions (dry run)")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of arm names")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    data["queries"] = [q for q in data["queries"] if q["relevant"]]
    if args.sessions:
        kept = {m["session_id"] for m in data["memories"][: args.sessions]}
        data["memories"] = data["memories"][: args.sessions]
        data["queries"] = [q for q in data["queries"] if set(q.get("relevant") or ()) & kept]
    if args.limit:
        data["queries"] = data["queries"][: args.limit]

    intents = [plan_query(q["query"], q.get("options")).intent for q in data["queries"]]
    census: dict[str, int] = {}
    for name in intents:
        census[name] = census.get(name, 0) + 1
    print(f"queries {len(data['queries'])}  intent census {census}")
    if not census.get("debug"):
        print("WARNING: no debug-intent queries; the debug table is never read")

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    print(
        f"config: cap={settings.max_evidence_per_session} "
        f"max_sessions={settings.evidence_max_sessions} gate={settings.min_evidence_score} "
        f"dense={settings.dense_enabled} rerank={settings.rerank_enabled} "
        f"position={settings.evidence_position_weight} promote={settings.evidence_operative_promotion}"
    )

    names = [a for a in args.arms.split(",") if a.strip()]
    container = Container(settings)
    rows_by_arm: dict[str, list[dict]] = {}
    try:
        build(container, data)
        print(f"corpus built: {len(data['memories'])} sessions", file=sys.stderr)
        for name in names:
            if name not in ARMS:
                print(f"unknown arm {name}", file=sys.stderr)
                continue
            ev.INTENT_KIND_BONUS = {k: dict(v) for k, v in ARMS[name].items()}
            print(f"--- pass: {name} ---", file=sys.stderr)
            rows_by_arm[name] = analyse(container, data, args.k)
    finally:
        ev.INTENT_KIND_BONUS = SHIPPED
        container.close()

    baseline_name = names[0]
    base_rows = rows_by_arm[baseline_name]
    print()
    print("=" * 72)
    print(f"baseline arm: {baseline_name}")
    print("=" * 72)
    base_summary = report(base_rows)

    out: dict[str, object] = {"baseline": baseline_name, "intent_census": census, "arms": {}}
    base_idx = index(base_rows)
    base_detail = detail_index(base_rows)
    for name, rows in rows_by_arm.items():
        summary = report(rows) if name != baseline_name else base_summary
        arm: dict[str, object] = {"headline": summary}
        if name != baseline_name:
            idx = index(rows)
            arm["paired"] = [paired(base_idx, idx, f) for f in OUTCOMES]
            arm["rank_shifts"] = shifts(base_detail, detail_index(rows))
        out["arms"][name] = arm

        if name != baseline_name:
            sh = arm["rank_shifts"]
            print()
            print(
                f"### rank sensitivity {baseline_name} -> {name}: "
                f"{sh['entries_moved']}/{sh['entries_compared']} dumped entries changed rank"
                + (
                    f", mean |Δrank| {sh['mean_abs_rank_change_moved']}"
                    if sh["entries_moved"]
                    else " (the arms are IDENTICAL on this corpus -- lever cannot be evaluated)"
                )
            )
            for kind, moved in sorted((sh.get("moved_by_kind") or {}).items()):
                print(
                    f"    {kind:<11} moved {moved['n']}  "
                    f"(up {moved['improved']} / down {moved['worsened']})"
                )
            print(f"### paired: {baseline_name} -> {name}")
            for res in arm["paired"]:
                arrow = "+" if res["arm_rate"] > res["baseline_rate"] else (
                    "-" if res["arm_rate"] < res["baseline_rate"] else "="
                )
                print(
                    f"  {res['field']:<15} {res['baseline_rate']:.4f} -> {res['arm_rate']:.4f} "
                    f"{arrow}  gained {res['gained']} / lost {res['lost']}  p={res['p']}"
                )

    if args.out:
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
