"""Phase B on the menu dump: digest composition, relay diagnostic.

The 11 llm-judgment misses (shortlisted, admissible, not picked) are
summary quality. The shipped digest shows each session's files, its first
280 characters, and its TWO LONGEST pooled chunks — a placeholder
heuristic (service.py sorts by length), not a relevance choice. This
replay re-runs the selection call on the recorded misses under three
digest variants, paired within query, same 8 sessions, same model,
temperature 0:

  V0 shipped    longest-2 chunks — the control; reproducing the recorded
                miss under V0 measures relay jitter, everything above it
                is the digest's doing;
  V1 scored     same shape, chunks BY FINAL SCORE instead of by length;
  V2 full       scored chunks + recorded-cause sentences + the session's
                last pooled line + session dates, and the query's key
                identifiers (plan entities/keywords) in the prompt.

Guard against the non-monotonicity trap: four currently-REACHED queries
run under V2 as well — a digest that fixes misses by unfixing correct
picks is a loss (the stack lesson, applied to digests).

Relay calls: 15 queries x 3 variants = 45 gpt-4o-mini calls, ~1 s each.

Usage::

    PYTHONPATH=src python scripts/digest_replay.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MENU = 8
CONTROL_N = 4


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        import os

        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def chat(system: str, user: str, model: str, base_url: str, key: str) -> str | None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=key, timeout=30.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=24,
    )
    return response.choices[0].message.content


def _date(ts: int | None) -> str:
    if not ts:
        return "?"
    try:
        t = ts / 1000 if ts > 1e11 else ts  # corpus stamps are ms sometimes
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return "?"


def digest_v0(s: dict) -> str:
    files = ", ".join(s["files"][:6]) if s["files"] else "(none)"
    chunks = "\n".join(f"    chunk: {t[:200]}" for t in s["longest"][:2])
    return (
        f"    files: {files}\n"
        f"    opening: {s['first'][:280]}\n"
        f"{chunks}"
    )


def digest_v1(s: dict) -> str:
    files = ", ".join(s["files"][:6]) if s["files"] else "(none)"
    chunks = "\n".join(
        f"    chunk: {c['text'][:200]}" for c in s["top_chunks"][:2]
    )
    return (
        f"    files: {files}\n"
        f"    opening: {s['first'][:280]}\n"
        f"{chunks}"
    )


def digest_v2(s: dict) -> str:
    files = ", ".join(s["files"][:6]) if s["files"] else "(none)"
    parts = [
        f"    files: {files}",
        f"    dates: {_date(s['ts_min'])} .. {_date(s['ts_max'])}",
        f"    opening: {s['first'][:280]}",
    ]
    parts += [
        f"    chunk: {c['text'][:200]}" for c in s["top_chunks"][:2]
    ]
    parts += [
        f"    recorded-cause: {c['text'][:200]}"
        for c in [{"text": c["text"]} for c in s.get("cause", [])][:2]
    ]
    if s.get("last"):
        parts.append(f"    last line: {s['last'][:200]}")
    return "\n".join(parts)


SYSTEM = (
    "You select which past engineering sessions recorded the cause or the "
    "fix of a described problem. Reply with exactly two numbers."
)


def build_user(row: dict, blocks: list[str], with_plan: bool) -> str:
    head = f"Problem / issue:\n{row['question'][:800]}\n"
    if with_plan:
        ents = sorted({
            v for vals in row["plan"]["entities"].values() for v in vals if v
        })[:12]
        if ents:
            head += f"\nKey identifiers mentioned in the problem: {', '.join(ents)}\n"
    return (
        head + "\nCandidate sessions:\n" + "\n".join(blocks) + "\n\n"
        "Which TWO sessions record the cause or the fix of this "
        "problem? Reply with the two numbers."
    )


def main() -> int:
    import os

    load_env()
    base_url = os.environ.get("CODEMEM_LLM_BASE_URL")
    key = os.environ.get("CODEMEM_LLM_API_KEY")
    model = os.environ.get("CODEMEM_LLM_MODEL", "gpt-4o-mini")
    if not (base_url and key):
        print("error: relay credentials missing (.env)", file=sys.stderr)
        return 2

    dump = json.loads(
        (ROOT / "eval/results/menu_dump.json").read_text(encoding="utf-8")
    )
    rows = dump["rows"]

    missed = [r for r in rows if r["answer_session"] and not r["sel_shown"]]
    judgment = [
        r for r in missed
        if r["answer_session"] in {s["sid"] for s in r["sessions"][:MENU]}
    ]
    reached_in_menu = [
        r for r in rows
        if r["answer_session"] and r["sel_shown"]
        and r["answer_session"] in {s["sid"] for s in r["sessions"][:MENU]}
    ]
    rng = random.Random(20261005)
    controls = rng.sample(reached_in_menu, min(CONTROL_N, len(reached_in_menu)))
    print(f"judgment misses: {len(judgment)} (archived: 11); "
          f"controls: {len(controls)}")

    variants = {
        "V0-shipped": (digest_v0, False),
        "V1-scored": (digest_v1, False),
        "V2-full": (digest_v2, True),
    }

    results: list[dict] = []
    for row in judgment + controls:
        menu = row["sessions"][:MENU]
        for vname, (fn, with_plan) in variants.items():
            blocks = [
                f"[{i}] " + fn(s) for i, s in enumerate(menu, start=1)
            ]
            user = build_user(row, blocks, with_plan)
            try:
                reply = chat(SYSTEM, user, model, base_url, key) or ""
            except Exception as exc:  # relay failure is a finding, not a crash
                print(f"  {row['query_id']} {vname}: RELAY ERROR {exc}")
                reply = ""
            numbers = [int(n) for n in re.findall(r"\d+", reply)]
            picked = [
                menu[n - 1]["sid"] for n in numbers if 1 <= n <= len(menu)
            ]
            picked = list(dict.fromkeys(picked))[:2]
            hit = row["answer_session"] in picked
            results.append({
                "query_id": row["query_id"],
                "group": "miss" if row in judgment else "control",
                "variant": vname,
                "picked": picked,
                "answer_picked": hit,
            })
            print(f"  {row['query_id']:52s} {vname:10s} "
                  f"{'HIT ' if hit else 'miss'} picked={[p[:8] for p in picked]}")

    print("\nsummary:")
    for vname in variants:
        for group in ("miss", "control"):
            sub = [r for r in results if r["variant"] == vname and r["group"] == group]
            if sub:
                print(f"  {vname:10s} {group:8s}: {sum(r['answer_picked'] for r in sub)}"
                      f"/{len(sub)} answer sessions picked")

    out = ROOT / "eval/results/digest_replay.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
