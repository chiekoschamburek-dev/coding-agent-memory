"""Is my own pipeline deterministic?

A/B comparisons are only meaningful if identical inputs produce identical
outputs. Two runs of the same configuration produced 0.333 and 0.367, and the
answer model is deterministic at temperature=0 (verified: 8/8 identical), so the
variance must come from our side.

This runs the exact same question twice in one process and diffs the prompt that
would be sent to the answer model, then does the same across two processes via a
hash digest. Any difference localises the nondeterminism.

Run:  PYTHONPATH=src python scripts/diagnose_determinism.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
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


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_client(settings: Settings):
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    return app, client


def main() -> int:
    qa = json.loads(pathlib.Path("eval/data/qa_modified.json").read_text(encoding="utf-8"))
    bench = json.loads(pathlib.Path("eval/data/benchmark.json").read_text(encoding="utf-8"))
    question = qa["questions"][0]

    digests: list[str] = []
    orders: list[str] = []
    for trial in range(2):
        settings = Settings.from_env()
        settings.data_dir = pathlib.Path(tempfile.mkdtemp())
        app, client = build_client(settings)
        try:
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
            response = client.post(
                "/search",
                json={
                    "query": question["question"],
                    "options": question["options"],
                    "user_id": f"bench:{question['repo']}",
                    "top_k": 100,
                },
            )
            data = response.json()["data"]
            blob = "\n\n".join(f"[{i}] {item['content']}" for i, item in enumerate(data, 1))
            digests.append(digest(blob))
            orders.append(",".join(item["id"] for item in data[:10]))
        finally:
            client.__exit__(None, None, None)

    print(f"same process, two independent builds:")
    print(f"  evidence digest trial 1: {digests[0]}")
    print(f"  evidence digest trial 2: {digests[1]}")
    print(f"  identical: {digests[0] == digests[1]}")
    if digests[0] != digests[1]:
        print(f"  top-10 order trial 1: {orders[0]}")
        print(f"  top-10 order trial 2: {orders[1]}")
        # Where do they first diverge?
        a = orders[0].split(",")
        b = orders[1].split(",")
        for index, (x, y) in enumerate(zip(a, b), 1):
            if x != y:
                print(f"  first divergence at rank {index}: {x} vs {y}")
                break
    else:
        print("  pipeline is deterministic in-process; run a second process to")
        print("  check for hash-seed dependent ordering.")

    print()
    print(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED', '(unset -> randomised)')}")
    print("  Hash randomisation changes set/dict iteration order across processes.")
    print("  Any ranking that breaks ties by iteration order is therefore")
    print("  nondeterministic run to run, which would make A/B results unreadable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
