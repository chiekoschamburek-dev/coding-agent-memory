"""Is an isolated edit/diff chunk useful, or does it only say "a file changed"?

Motivation
----------
Inside one message the chunker splits structural segments (a diff with a header,
a stacktrace) away from the prose around them, so a returned item can be the
change alone. The question a reader asks is whether that item is useful without
its context. This script measures the two halves of that:

A. **Corpus census (static).** Among stored chunks that carry an operative line
   ("this file was changed"), what structural kind are they, how much *prose* do
   they carry (a tool-call JSON record has none), and do they share their session
   with at least one non-operative chunk — i.e. does the explanation exist
   anywhere in the session, or is the session itself a single bare record?

B. **Payload co-return (per question).** The answer model reads the returned
   payload, not the store. When the gold session's operative chunk is returned,
   how many same-session companion items come with it, how many of those carry
   prose (real explanation rather than another bare record), in what order, and
   does the operative item itself carry ``old_string``/``new_string`` (context
   that travels inside the tool call)?

``gold_op_present`` here is deliberately the same predicate as ``run_evidence``'s
``decisive_present`` (identical parser), so the two scripts must agree on the
same run. If they diverge, this script is wrong.

Run::

    PYTHONPATH=src python scripts/diagnose_chunk_context.py
    PYTHONPATH=src python scripts/diagnose_chunk_context.py --json eval/results/chunk_context.json
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

sys.path.insert(0, "src")

logging.disable(logging.WARNING)

# Byte-for-byte the marker set from eval/run_evidence.py, so the two agree.
# Note the tool pattern deliberately has NO re.DOTALL: the payload is one line,
# and letting `.*` span lines makes json.loads fail and silently drops file_path.
_OPERATIVE_RES = (
    re.compile(r"\[tool (?:Edit|Write|MultiEdit|NotebookEdit)\]\s*(\{.*)"),
    re.compile(r"^diff --git a/(\S+) b/"),
    re.compile(r"^\+\+\+ b/(\S+)"),
    re.compile(r"^--- a/(\S+)"),
    re.compile(r"The file (\S+) has been updated"),
)
_BACKSLASH = chr(92)

# Lines that are JSON/patch mechanics, not prose. A line counts as prose only if
# it survives all of these filters and still has >= 5 alphabetic words.
_JSONISH_START = ("{", "}", "[", "]", '"', "'")
_DIFFISH_START = ("+", "-", "@@", "---", "+++", "=")
_KEYVALUE_RE = re.compile(r"^[\w.\-]+\s*[:=]\s")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _basename(path: str) -> str:
    return path.replace(_BACKSLASH, "/").rstrip("`'\".,;").rsplit("/", 1)[-1]


def modified_basenames(blob: str) -> set[str]:
    """Files shown as MODIFIED, by tool call, update notice, or diff header."""
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


def has_operative_line(text: str) -> bool:
    return any(p.search(line) for line in text.splitlines() for p in _OPERATIVE_RES)


def carries_patch_context(text: str) -> bool:
    """True when the item's own text holds before/after content (Edit payload)."""
    return '"old_string"' in text or '"new_string"' in text


def prose_words(text: str) -> int:
    """Count natural-language words, ignoring JSON/patch mechanics.

    A bare tool-call record scores 0; an explanatory paragraph scores high. This
    is a heuristic, so a couple of examples are printed for inspection.
    """
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(_JSONISH_START):
            continue
        if stripped.startswith(_DIFFISH_START) or _KEYVALUE_RE.match(stripped):
            continue
        if any(p.search(stripped) for p in _OPERATIVE_RES):
            continue
        words = _WORD_RE.findall(stripped)
        if len(words) >= 5:
            total += len(words)
    return total


def run(questions_path: pathlib.Path, benchmark_path: pathlib.Path, top_k: int,
        examples: int, settings_overrides: dict) -> dict:
    from fastapi.testclient import TestClient

    from codemem.api.app import create_app
    from codemem.core.config import Settings

    qa = json.loads(questions_path.read_text(encoding="utf-8"))
    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = qa["questions"]

    settings = Settings.from_env()
    settings.data_dir = pathlib.Path(tempfile.mkdtemp())
    for key, value in settings_overrides.items():
        setattr(settings, key, value)
    app = create_app(settings)

    with TestClient(app) as client:
        for memory in bench["memories"]:
            client.post(
                "/add",
                json={
                    "request_id": f"cc:{memory['id']}",
                    "user_id": memory["user_id"],
                    "session_id": memory["session_id"],
                    "messages": memory["messages"],
                },
            )

        store = app.state.container.store
        with store._read() as conn:  # noqa: SLF001
            rows = list(
                conn.execute(
                    "SELECT m.id, m.session_id, c.kind AS chunk_kind, m.text "
                    "FROM memory m LEFT JOIN chunk c ON c.id = m.chunk_id"
                )
            )
        memories = {
            f"mem_{r['id']}": {
                "session_id": r["session_id"],
                "chunk_kind": r["chunk_kind"] or "?",
                "text": r["text"],
            }
            for r in rows
        }

        # ---------------------------------------------------------------- A --
        session_has_op: dict[str, bool] = {}
        session_has_nonop: dict[str, bool] = {}
        op_kinds: dict[str, int] = {}
        op_total = 0
        op_bare = 0
        op_with_patch_context = 0
        bare_examples: list[dict] = []
        for mem_id, mem in memories.items():
            op = has_operative_line(mem["text"])
            session_has_op[mem["session_id"]] = (
                session_has_op.get(mem["session_id"], False) or op
            )
            if not op:
                session_has_nonop[mem["session_id"]] = True
                continue
            op_total += 1
            op_kinds[mem["chunk_kind"]] = op_kinds.get(mem["chunk_kind"], 0) + 1
            if prose_words(mem["text"]) == 0:
                op_bare += 1
                if len(bare_examples) < examples:
                    bare_examples.append(
                        {"memory_id": mem_id, "chunk_kind": mem["chunk_kind"],
                         "chars": len(mem["text"]), "head": mem["text"][:240]}
                    )
            if carries_patch_context(mem["text"]):
                op_with_patch_context += 1

        op_sessions = {s for s, has in session_has_op.items() if has}
        op_sessions_with_context = {s for s in op_sessions if session_has_nonop.get(s)}
        op_sessions_single = op_sessions - op_sessions_with_context

        # ---------------------------------------------------------------- B --
        per_question: list[dict] = []
        for question in questions:
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
            gold = _basename(question["gold_file"])
            gold_session = question.get("answer_session")

            gold_op_index = None
            for index, item in enumerate(data):
                if gold in modified_basenames(item["content"]):
                    gold_op_index = index
                    break

            companion_total = 0
            companion_with_prose = 0
            companion_operative = 0
            companion_before = 0
            companion_after = 0
            if gold_op_index is not None:
                for index, item in enumerate(data):
                    if index == gold_op_index:
                        continue
                    info = memories.get(item["id"])
                    if not info or info["session_id"] != gold_session:
                        continue
                    companion_total += 1
                    if prose_words(item["content"]) > 0:
                        companion_with_prose += 1
                    if has_operative_line(item["content"]):
                        companion_operative += 1
                    if index < gold_op_index:
                        companion_before += 1
                    else:
                        companion_after += 1

            stored_same_session = sum(
                1 for mem in memories.values() if mem["session_id"] == gold_session
            )
            stored_nonop_same_session = sum(
                1
                for mem in memories.values()
                if mem["session_id"] == gold_session
                and not has_operative_line(mem["text"])
            )

            op_item = data[gold_op_index] if gold_op_index is not None else None
            op_info = memories.get(op_item["id"]) if op_item else None
            per_question.append(
                {
                    "query_id": question["query_id"],
                    "gold_op_present": gold_op_index is not None,
                    "gold_op_rank": gold_op_index,
                    "gold_op_chunk_kind": op_info["chunk_kind"] if op_info else None,
                    "gold_op_carries_patch_context": (
                        carries_patch_context(op_item["content"]) if op_item else None
                    ),
                    "companions_total": companion_total,
                    "companions_with_prose": companion_with_prose,
                    "companions_operative_records": companion_operative,
                    "companions_before": companion_before,
                    "companions_after": companion_after,
                    "stored_same_session": stored_same_session,
                    "stored_nonoperative_same_session": stored_nonop_same_session,
                }
            )

    present = [r for r in per_question if r["gold_op_present"]]
    n = len(per_question)
    n_present = len(present)

    def mean(key: str) -> float:
        return sum(r[key] for r in present) / n_present if n_present else 0.0

    def rate(key: str) -> float:
        return sum(1 for r in present if r[key]) / n_present if n_present else 0.0

    return {
        "n_questions": n,
        "A_operative_chunks": {
            "total": op_total,
            "kinds": dict(sorted(op_kinds.items(), key=lambda kv: -kv[1])),
            "bare_no_prose": op_bare,
            "with_patch_context": op_with_patch_context,
            "sessions_with_operative": len(op_sessions),
            "sessions_that_also_hold_nonoperative": len(op_sessions_with_context),
            "sessions_whose_only_chunk_is_operative": len(op_sessions_single),
        },
        "B_payload": {
            "gold_op_present_rate": n_present / n if n else 0.0,
            "mean_companions": mean("companions_total"),
            "mean_companions_with_prose": mean("companions_with_prose"),
            "bare_no_companion_rate": (
                sum(1 for r in present if r["companions_total"] == 0) / n_present
                if n_present else 0.0
            ),
            "no_prose_companion_rate": (
                sum(1 for r in present if r["companions_with_prose"] == 0) / n_present
                if n_present else 0.0
            ),
            "gold_op_carries_patch_context_rate": rate("gold_op_carries_patch_context"),
            "mean_stored_nonoperative_in_gold_session": mean(
                "stored_nonoperative_same_session"
            ),
        },
        "examples": bare_examples,
        "rows": per_question,
        "settings": {
            "dense_enabled": settings.dense_enabled,
            "rerank_enabled": settings.rerank_enabled,
            "evidence_full_count": settings.evidence_full_count,
            "evidence_item_tokens": settings.evidence_item_tokens,
            "evidence_max_sessions": settings.evidence_max_sessions,
            "max_evidence_per_session": settings.max_evidence_per_session,
            "evidence_operative_promotion": settings.evidence_operative_promotion,
        },
    }


def report(result: dict) -> None:
    a = result["A_operative_chunks"]
    b = result["B_payload"]
    s = result["settings"]
    print()
    print("=" * 68)
    print("Chunk context: is an isolated edit/diff chunk useful?")
    print("=" * 68)
    print(f"settings: dense={s['dense_enabled']}, rerank={s['rerank_enabled']}, "
          f"cap={s['max_evidence_per_session']}, sessions={s['evidence_max_sessions']}, "
          f"promotion={s['evidence_operative_promotion']}, "
          f"item_tokens={s['evidence_item_tokens']}, full_count={s['evidence_full_count']}")
    print()
    print("A. Corpus census (static)")
    print(f"  operative chunks in store        : {a['total']}")
    print(f"  ... with no prose at all         : {a['bare_no_prose']} "
          f"({a['bare_no_prose'] / a['total']:.0%})")
    print(f"  ... carrying old/new_string      : {a['with_patch_context']} "
          f"({a['with_patch_context'] / a['total']:.0%})")
    print(f"  structural kinds                 : {a['kinds']}")
    print(f"  sessions holding >=1 operative   : {a['sessions_with_operative']}")
    print(f"  ... also holding a non-operative : "
          f"{a['sessions_that_also_hold_nonoperative']} "
          f"({a['sessions_that_also_hold_nonoperative'] / a['sessions_with_operative']:.0%})")
    print(f"  ... whose ONLY chunk is operative: "
          f"{a['sessions_whose_only_chunk_is_operative']}")
    print()
    print("B. Payload co-return (per question, what the answer model reads)")
    print(f"  questions                        : {result['n_questions']}")
    print(f"  gold operative item returned     : {b['gold_op_present_rate']:.3f}")
    print(f"  mean same-session companions     : {b['mean_companions']:.2f}")
    print(f"  ... of which carry prose         : {b['mean_companions_with_prose']:.2f}")
    print(f"  NO companion at all              : {b['bare_no_companion_rate']:.3f}")
    print(f"  NO prose companion (bare record) : {b['no_prose_companion_rate']:.3f}")
    print(f"  operative item carries old/new_string: "
          f"{b['gold_op_carries_patch_context_rate']:.3f}")
    print(f"  mean non-operative chunks stored in the gold session: "
          f"{b['mean_stored_nonoperative_in_gold_session']:.1f}")
    print()
    if result["examples"]:
        print("  Example 'no prose' operative chunks (heuristic sanity check):")
        for ex in result["examples"]:
            print(f"    [{ex['memory_id']} kind={ex['chunk_kind']} {ex['chars']} chars] "
                  f"{ex['head']!r}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=pathlib.Path,
                        default=pathlib.Path("eval/data/qa_modified.json"))
    parser.add_argument("--data", type=pathlib.Path,
                        default=pathlib.Path("eval/data/benchmark.json"))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--cap", type=int, default=None,
                        help="override max_evidence_per_session")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--operative-promotion", type=int, default=None)
    parser.add_argument("--item-tokens", type=int, default=None)
    parser.add_argument("--full-count", type=int, default=None)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    env_file = pathlib.Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

    if not args.qa.exists() or not args.data.exists():
        print("error: build the datasets first "
              "(eval/build_benchmark.py, build_qa_modified.py)", file=sys.stderr)
        return 2

    overrides: dict = {}
    if args.cap is not None:
        overrides["max_evidence_per_session"] = args.cap
    if args.max_sessions is not None:
        overrides["evidence_max_sessions"] = args.max_sessions
    if args.operative_promotion is not None:
        overrides["evidence_operative_promotion"] = args.operative_promotion
    if args.item_tokens is not None:
        overrides["evidence_item_tokens"] = args.item_tokens
    if args.full_count is not None:
        overrides["evidence_full_count"] = args.full_count

    result = run(args.qa, args.data, top_k=args.top_k, examples=args.examples,
                 settings_overrides=overrides)
    report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
