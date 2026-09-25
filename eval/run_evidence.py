"""Deterministic evidence-sufficiency evaluation.

Why not an end-to-end accuracy number
-------------------------------------
The rules assign Add and Search to us and Answer/Eval to the platform, which uses
its own locked answer model and prompt. Measuring answer accuracy with a local
proxy therefore (a) evaluates the platform's half, and (b) depends on a model we
do not control. Measured on our relay, the same prompt at temperature=0 returned
"B" four times and "C" four times out of eight, and an explicit ``seed`` did not
stabilise it — so a single-pass accuracy number is a coin flip, not a result.

What we are actually responsible for is this: the returned ``data[]`` is ranked,
denoised, and its ``content`` carries the evidence the answer model will read in
that order. That is measurable exactly, with no model in the loop.

The measurement
---------------
For a question whose answer is recorded in the trajectory, check whether the
**decisive evidence** survived into what we return:

* *decisive present* — the returned content contains a line associating the gold
  artifact with the operative action (a tool call, an update notice, or a diff
  header);
* *ambiguous* — such a line also appears for a distractor, making the returned
  evidence unable to settle the question;
* *decidable* — decisive present and not ambiguous. This is the number to raise.

All three are computed by parsing strings, so they are reproducible, fast, and
free, and they improve only when our retrieval, ordering, or content selection
improves.

Under a token prefix
--------------------
The platform feeds the answer model a *token-counted prefix* of ``data[]`` in our
order, so evidence sitting in item 60 of a 67-item payload is not evidence the
answer model ever sees. Scoring the concatenation of everything we return
credits exactly that, and inflates long lists: a policy that returns more items
can only look better, never worse, which is the wrong gradient for a denoising
system. So every rate is additionally measured at a set of token budgets, taking
items in order and cutting the straddling one mid-way. Those rows are the graded
object; the unlimited row is kept for comparison with earlier runs.

Run::

    PYTHONPATH=src python eval/run_evidence.py
    PYTHONPATH=src python eval/run_evidence.py --json eval/results/evidence.json
    PYTHONPATH=src python eval/run_evidence.py --prefix-tokens 2000 4000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, "src")

# Markers that record an operative action: "this file was changed".
_OPERATIVE_RES = (
    re.compile(r"\[tool (?:Edit|Write|MultiEdit|NotebookEdit)\]\s*(\{.*)"),
    re.compile(r"^diff --git a/(\S+) b/"),
    re.compile(r"^\+\+\+ b/(\S+)"),
    re.compile(r"^--- a/(\S+)"),
    re.compile(r"The file (\S+) has been updated"),
)

_BACKSLASH = chr(92)


def _basename(path: str) -> str:
    return path.replace(_BACKSLASH, "/").rstrip("`'\".,;").rsplit("/", 1)[-1]


def modified_basenames(blob: str) -> set[str]:
    """Files shown as modified in a returned-content blob.

    Parses the ``file_path`` field of tool-call payloads rather than
    substring-searching for the file name: an Edit payload's ``old_string`` can
    name other files, which inflated an earlier version of this measurement from
    30% to 45%.
    """
    out: set[str] = set()
    for line in blob.splitlines():
        tool = _OPERATIVE_RES[0].search(line)
        if tool:
            try:
                payload = json.loads(tool.group(1))
            except (json.JSONDecodeError, ValueError):
                payload = {}
            path = payload.get("file_path") if isinstance(payload, dict) else None
            if isinstance(path, str):
                out.add(_basename(path))
        for pattern in _OPERATIVE_RES[1:]:
            match = pattern.search(line)
            if match:
                out.add(_basename(match.group(1)))
    return out


def _prefix_view(contents: list[str], costs: list[int], budget: int) -> tuple[str, int, int]:
    """The content the answer model actually receives: a token-counted prefix.

    Returns ``(blob, n_items_visible, tokens_visible)``. Items are taken in the
    order we emit them, and the one that straddles the budget is cut mid-way —
    modelling the cut as item-aligned would credit the tail of a long item that
    the model never receives, which is the same inflation in miniature.

    ``budget <= 0`` means no limit.
    """
    from codemem.core.tokens import truncate_to_tokens

    if budget <= 0:
        return "\n\n".join(contents), len(contents), sum(costs)
    parts: list[str] = []
    used = 0
    visible = 0
    for content, cost in zip(contents, costs):
        if used + cost <= budget:
            parts.append(content)
            used += cost
            visible += 1
            continue
        remaining = budget - used
        if remaining > 0:
            parts.append(truncate_to_tokens(content, remaining))
            used = budget
            visible += 1
        break
    return "\n\n".join(parts), visible, used


def run(questions_path: Path, benchmark_path: Path, *, top_k: int, limit: int | None,
        settings_overrides: dict, quiet: bool,
        prefix_budgets: Sequence[int] = ()) -> dict:
    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient

    from codemem.api.app import create_app
    from codemem.core.config import Settings
    from codemem.core.tokens import count_tokens

    qa = json.loads(questions_path.read_text(encoding="utf-8"))
    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = qa["questions"]
    if limit:
        questions = questions[:limit]

    settings = Settings.from_env()
    settings.data_dir = Path(tempfile.mkdtemp())
    for key, value in settings_overrides.items():
        setattr(settings, key, value)
    app = create_app(settings)

    rows: list[dict] = []
    with TestClient(app) as client:
        for memory in bench["memories"]:
            client.post(
                "/add",
                json={
                    "request_id": f"ev:{memory['id']}",
                    "user_id": memory["user_id"],
                    "session_id": memory["session_id"],
                    "messages": memory["messages"],
                },
            )

        store = app.state.container.store
        with store._read() as conn:  # noqa: SLF001
            session_of = {
                f"mem_{r['id']}": r["session_id"]
                for r in conn.execute("SELECT id, session_id FROM memory")
            }

        for index, question in enumerate(questions, 1):
            response = client.post(
                "/search",
                json={
                    "query": question["question"],
                    "options": question["options"],
                    "user_id": f"bench:{question['repo']}",
                    "top_k": top_k,
                },
            )
            data = response.json().get("data", [])
            contents = [item["content"] for item in data]
            costs = [count_tokens(content) for content in contents]

            gold = _basename(question["gold_file"])
            distractors = [
                _basename(option)
                for i, option in enumerate(question["options"])
                if i != question["gold_index"]
            ]

            def judge(blob: str) -> tuple[bool, bool]:
                modified = modified_basenames(blob)
                return (
                    gold in modified,
                    any(distractor in modified for distractor in distractors),
                )

            decisive, distractor_hit = judge("\n\n".join(contents))
            shown_sessions = {session_of.get(item["id"]) for item in data}
            answer_session = question.get("answer_session")

            by_prefix: dict[str, dict] = {}
            for budget in prefix_budgets:
                if budget <= 0:
                    continue
                prefix_blob, visible, tokens = _prefix_view(contents, costs, budget)
                p_decisive, p_distractor = judge(prefix_blob)
                by_prefix[str(budget)] = {
                    "decisive_present": p_decisive,
                    "ambiguous": p_distractor,
                    "decidable": p_decisive and not p_distractor,
                    "n_visible": visible,
                    "tokens": tokens,
                }

            rows.append(
                {
                    "query_id": question["query_id"],
                    "decisive_present": decisive,
                    "ambiguous": distractor_hit,
                    "decidable": decisive and not distractor_hit,
                    "n_returned": len(data),
                    "n_sessions": len(shown_sessions - {None}),
                    "session_retrieved": (
                        answer_session in shown_sessions if answer_session else None
                    ),
                    "tokens": sum(costs),
                    "by_prefix": by_prefix,
                }
            )
            if not quiet and index % 20 == 0:
                rate = sum(r["decidable"] for r in rows) / len(rows)
                print(f"    {index}/{len(questions)} decidable={rate:.3f}", flush=True)

    n = len(rows)

    def rate(key: str) -> float:
        return sum(bool(r[key]) for r in rows) / n if n else 0.0

    prefix_summary = {}
    for budget in sorted({int(b) for b in prefix_budgets if int(b) > 0}):
        entries = [r["by_prefix"].get(str(budget)) for r in rows]
        entries = [e for e in entries if e is not None]
        if not entries:
            continue
        m = len(entries)
        prefix_summary[str(budget)] = {
            "decisive_present_rate": sum(e["decisive_present"] for e in entries) / m,
            "ambiguity_rate": sum(e["ambiguous"] for e in entries) / m,
            "decidable_rate": sum(e["decidable"] for e in entries) / m,
            "mean_visible": sum(e["n_visible"] for e in entries) / m,
            "mean_tokens": sum(e["tokens"] for e in entries) / m,
        }

    return {
        "n": n,
        "session_retrieved_rate": rate("session_retrieved"),
        "decisive_present_rate": rate("decisive_present"),
        "ambiguity_rate": rate("ambiguous"),
        "decidable_rate": rate("decidable"),
        "mean_returned": sum(r["n_returned"] for r in rows) / n if n else 0.0,
        "mean_sessions": sum(r["n_sessions"] for r in rows) / n if n else 0.0,
        "mean_tokens": sum(r["tokens"] for r in rows) / n if n else 0.0,
        "prefix": prefix_summary,
        "rows": rows,
        "settings": settings_overrides,
    }


def report(result: dict) -> None:
    print()
    print("=" * 68)
    print("Evidence sufficiency (deterministic; no model in the loop)")
    print("=" * 68)
    print(f"questions                : {result['n']}")
    print(f"answer session retrieved : {result['session_retrieved_rate']:.3f}")
    print(f"mean items returned      : {result['mean_returned']:.1f}")
    print(f"mean sessions returned   : {result['mean_sessions']:.1f}")
    print(f"mean tokens returned     : {result['mean_tokens']:.0f}")
    print()
    print("  as graded: only the token prefix the answer model reads")
    print(f"  {'prefix':>8} | {'visible':>7} | {'tokens':>7} | {'decisive':>8} |"
          f" {'ambig':>6} | {'DECIDABLE':>9}")
    print("  " + "-" * 62)
    if result["prefix"]:
        for budget in sorted(result["prefix"], key=int):
            entry = result["prefix"][budget]
            print(
                f"  {budget:>8} | {entry['mean_visible']:>7.1f} |"
                f" {entry['mean_tokens']:>7.0f} | {entry['decisive_present_rate']:>8.3f} |"
                f" {entry['ambiguity_rate']:>6.3f} | {entry['decidable_rate']:>9.3f}"
            )
    unlimited = (
        f"  {'all':>8} | {result['mean_returned']:>7.1f} |"
        f" {result['mean_tokens']:>7.0f} | {result['decisive_present_rate']:>8.3f} |"
        f" {result['ambiguity_rate']:>6.3f} | {result['decidable_rate']:>9.3f}"
    )
    print(unlimited)
    print()
    print("  The 'all' row concatenates everything we return, so it credits evidence")
    print("  the platform never feeds the answer model — it is an upper bound, not a")
    print("  result. The prefix rows are what is graded.")
    print()
    print("  'Decidable' is the number to raise: the evidence the model can see")
    print("  contains the line that settles the question and no distractor contradicts")
    print("  it. It is fully reproducible, so a change either moves it or does not.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path("eval/data/qa_modified.json"))
    parser.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--item-tokens", type=int, default=None)
    parser.add_argument("--full-count", type=int, default=None)
    parser.add_argument("--ptr-tokens", type=int, default=None)
    parser.add_argument("--operative-weight", type=float, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--operative-promotion", type=int, default=None,
                        help="override evidence_operative_promotion (-1=all, 0=never, N=top N)")
    parser.add_argument("--cap", type=int, default=None,
                        help="override max_evidence_per_session")
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        nargs="+",
        default=[1000, 2000, 4000, 8000],
        help="token budgets to score at, modelling the platform's counted prefix",
    )
    parser.add_argument("--rows", type=Path, default=None,
                        help="dump per-question rows, for paired comparison")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

    if not args.qa.exists() or not args.data.exists():
        print("error: build the datasets first (eval/build_benchmark.py, build_qa_modified.py)",
              file=sys.stderr)
        return 2

    overrides: dict = {}
    if args.item_tokens is not None:
        overrides["evidence_item_tokens"] = args.item_tokens
    if args.full_count is not None:
        overrides["evidence_full_count"] = args.full_count
    if args.ptr_tokens is not None:
        overrides["evidence_ptr_tokens"] = args.ptr_tokens
    if args.operative_weight is not None:
        overrides["evidence_operative_weight"] = args.operative_weight
    if args.operative_promotion is not None:
        overrides["evidence_operative_promotion"] = args.operative_promotion
    if args.max_sessions is not None:
        overrides["evidence_max_sessions"] = args.max_sessions
    if args.cap is not None:
        overrides["max_evidence_per_session"] = args.cap

    result = run(args.qa, args.data, top_k=args.top_k, limit=args.limit,
                 settings_overrides=overrides, quiet=args.quiet,
                 prefix_budgets=args.prefix_tokens)
    report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(
                {k: v for k, v in result.items() if k != "rows"}, handle, indent=1
            )
        print(f"\nwrote {args.json}")

    if args.rows:
        # Per-question outcomes, so two runs can be compared as pairs rather
        # than as two independent rates: a three-question difference on n=30
        # is either three questions that moved or noise, and only the paired
        # view can tell which.
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        with args.rows.open("w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "query_id": r["query_id"],
                        "decidable": r["decidable"],
                        "decisive_present": r["decisive_present"],
                        "ambiguous": r["ambiguous"],
                    }
                    for r in result["rows"]
                ],
                handle,
                indent=1,
            )
        print(f"wrote {args.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
