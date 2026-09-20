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

Run::

    PYTHONPATH=src python eval/run_evidence.py
    PYTHONPATH=src python eval/run_evidence.py --json eval/results/evidence.json
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


def run(questions_path: Path, benchmark_path: Path, *, top_k: int, limit: int | None,
        settings_overrides: dict, quiet: bool) -> dict:
    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient

    from codemem.api.app import create_app
    from codemem.core.config import Settings

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
            blob = "\n\n".join(item["content"] for item in data)
            modified = modified_basenames(blob)

            gold = _basename(question["gold_file"])
            decisive = gold in modified
            distractor_hit = any(
                _basename(option) in modified
                for i, option in enumerate(question["options"])
                if i != question["gold_index"]
            )
            shown_sessions = {session_of.get(item["id"]) for item in data}
            answer_session = question.get("answer_session")

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
                }
            )
            if not quiet and index % 20 == 0:
                rate = sum(r["decidable"] for r in rows) / len(rows)
                print(f"    {index}/{len(questions)} decidable={rate:.3f}", flush=True)

    n = len(rows)
    return {
        "n": n,
        "session_retrieved_rate": (
            sum(1 for r in rows if r["session_retrieved"]) / n if n else 0.0
        ),
        "decisive_present_rate": sum(r["decisive_present"] for r in rows) / n if n else 0.0,
        "ambiguity_rate": sum(r["ambiguous"] for r in rows) / n if n else 0.0,
        "decidable_rate": sum(r["decidable"] for r in rows) / n if n else 0.0,
        "mean_returned": sum(r["n_returned"] for r in rows) / n if n else 0.0,
        "mean_sessions": sum(r["n_sessions"] for r in rows) / n if n else 0.0,
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
    print(f"decisive evidence present: {result['decisive_present_rate']:.3f}")
    print(f"ambiguous (distractor too): {result['ambiguity_rate']:.3f}")
    print(f"DECIDABLE                : {result['decidable_rate']:.3f}")
    print(f"mean items returned      : {result['mean_returned']:.1f}")
    print(f"mean sessions returned   : {result['mean_sessions']:.1f}")
    print()
    print("  'Decidable' is the number to raise: the returned evidence contains the")
    print("  line that settles the question and no distractor contradicts it. It is")
    print("  fully reproducible, so a change either moves it or does not.")


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
    if args.max_sessions is not None:
        overrides["evidence_max_sessions"] = args.max_sessions

    result = run(args.qa, args.data, top_k=args.top_k, limit=args.limit,
                 settings_overrides=overrides, quiet=args.quiet)
    report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(
                {k: v for k, v in result.items() if k != "rows"}, handle, indent=1
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
