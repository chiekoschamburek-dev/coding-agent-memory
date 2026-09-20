"""Why is accuracy 0.333 when the answer session is retrieved 100% of the time?

The decisive signal in these transcripts is the tool-call record:

    [tool Edit] {"file_path": ".../sklearn/mixture/base.py", ...}

It is what separates a file the session MODIFIED from one it merely READ. This
script measures whether that line survives into the returned evidence, and
whether any distractor is ever shown as edited (which would make the question
genuinely ambiguous).

Run:  PYTHONPATH=src python scripts/diagnose_edit_signal.py
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, "src")

for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()

logging.disable(logging.WARNING)

from fastapi.testclient import TestClient  # noqa: E402

from codemem.api.app import create_app  # noqa: E402
from codemem.core.config import Settings  # noqa: E402

# Match the tool marker plus its JSON payload, so the file_path FIELD is parsed
# rather than searched for as a substring. A substring search produced false
# positives, because an Edit payload's old_string/new_string can mention other
# files.
TOOL_RE = re.compile(r"\[tool (Edit|Write|MultiEdit|NotebookEdit)\]\s*(\{.*)", re.DOTALL)

BACKSLASH = chr(92)


# Three independent textual forms reveal that a file was MODIFIED. Counting only
# the tool-call marker undercounts how often the evidence is present, which would
# misattribute the bottleneck.
UPDATED_RE = re.compile(r"The file (\S+) has been updated")
DIFF_RE = re.compile(r"^diff --git a/(\S+) b/")


def pure(path: str) -> str:
    return path.replace(BACKSLASH, "/").rsplit("/", 1)[-1]


def edited_basenames(blob: str) -> set[str]:
    """Files shown as modified, by any of the three evidence forms."""
    out: set[str] = set()
    for line in blob.splitlines():
        match = TOOL_RE.search(line)
        if match:
            try:
                payload = json.loads(match.group(2))
            except (json.JSONDecodeError, ValueError):
                payload = {}
            path = payload.get("file_path")
            if isinstance(path, str):
                out.add(pure(path))
        updated = UPDATED_RE.search(line)
        if updated:
            out.add(pure(updated.group(1).strip("`'\".")))
        diff = DIFF_RE.match(line)
        if diff:
            out.add(pure(diff.group(1)))
    return out


def main(limit: int = 20) -> int:
    qa = json.loads(pathlib.Path("eval/data/qa_modified.json").read_text(encoding="utf-8"))
    bench = json.loads(pathlib.Path("eval/data/benchmark.json").read_text(encoding="utf-8"))
    questions = qa["questions"][:limit]

    settings = Settings.from_env()
    settings.data_dir = pathlib.Path(tempfile.mkdtemp())
    app = create_app(settings)

    with TestClient(app) as client:
        for memory in bench["memories"]:
            client.post(
                "/add",
                json={
                    "request_id": f"b:{memory['id']}",
                    "user_id": memory["user_id"],
                    "session_id": memory["session_id"],
                    "messages": memory["messages"],
                },
            )

        gold_signal = 0
        ambiguous = 0
        for question in questions:
            response = client.post(
                "/search",
                json={
                    "query": question["question"],
                    "options": question["options"],
                    "user_id": f"bench:{question['repo']}",
                    "top_k": 100,
                },
            )
            blob = "\n\n".join(item["content"] for item in response.json()["data"])
            edited = edited_basenames(blob)

            gold_name = question["gold_file"].rsplit("/", 1)[-1]
            if gold_name in edited:
                gold_signal += 1
            for index, option in enumerate(question["options"]):
                if index == question["gold_index"]:
                    continue
                if option.rsplit("/", 1)[-1] in edited:
                    ambiguous += 1
                    break

    n = len(questions)
    print(f"questions: {n}")
    print(
        f"  gold shown as MODIFIED (tool call, result, or diff): "
        f"{gold_signal}/{n} ({gold_signal / n:.0%})"
    )
    print(
        f"  a distractor ALSO shown as edited (ambiguous) : "
        f"{ambiguous}/{n} ({ambiguous / n:.0%})"
    )
    print()
    print("  Reading: retrieval succeeds (the answer session is in context 100% of")
    print("  the time), but the window we return usually drops the one line that")
    print("  settles the question. The bottleneck is evidence selection, not recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
