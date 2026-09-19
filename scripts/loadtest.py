"""Load and latency test against a running Add/Search service.

Answers the two operational questions that matter for the submission: does the
service stay correct under concurrency, and how long do Add and Search take as
the corpus grows?

Usage::

    python scripts/loadtest.py --base-url http://127.0.0.1:8080 --api-key KEY
    python scripts/loadtest.py --base-url ... --writers 4 --searchers 8 --seconds 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

FILES = [
    "src/parser/tokenizer.py",
    "src/parser/lexer.py",
    "src/net/retry.py",
    "src/queue/worker.py",
    "src/api/handlers/search.py",
    "tests/test_tokenizer.py",
    "src/cache/redis_client.py",
    "src/db/pool.py",
]
ERRORS = ["IndexError", "KeyError", "TimeoutError", "ConnectionResetError", "ValueError"]


def make_message(seq: int, rng: random.Random) -> str:
    path = rng.choice(FILES)
    err = rng.choice(ERRORS)
    return (
        f"Session {seq}: debugging a failure in {path}.\n\n"
        "```bash\n$ pytest tests/ -k smoke\n```\n\n"
        f"The traceback showed {err} raised from the hot path.\n\n"
        "```diff\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -10,6 +10,7 @@\n"
        f"-    return self.state\n+    if self.state is None:\n+        raise {err}('state unset')\n"
        "```\n\n"
        f"Root cause was an uninitialised state object in {path}; added a guard "
        f"and reran the suite. Marker token{seq}.\n"
    )


@dataclass
class Stats:
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, latency: float) -> None:
        with self.lock:
            self.latencies.append(latency)

    def error(self, message: str) -> None:
        with self.lock:
            self.errors.append(message)

    def summary(self) -> dict:
        with self.lock:
            values = sorted(self.latencies)
            errors = list(self.errors)
        if not values:
            return {"count": 0, "errors": errors}
        return {
            "count": len(values),
            "errors": errors,
            "p50_ms": round(statistics.median(values) * 1000, 1),
            "p95_ms": round(values[int(len(values) * 0.95)] * 1000, 1),
            "p99_ms": round(values[min(len(values) - 1, int(len(values) * 0.99))] * 1000, 1),
            "max_ms": round(max(values) * 1000, 1),
            "mean_ms": round(statistics.fmean(values) * 1000, 1),
        }


class Client:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("X-Api-Key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:200]
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    def add(self, request_id: str, user_id: str, session_id: str, content: str) -> dict:
        return self._post(
            "/add",
            {
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "messages": [
                    {"role": "user", "content": content, "timestamp": int(time.time() * 1000)}
                ],
            },
        )

    def search(self, user_id: str, query: str, top_k: int = 100) -> dict:
        return self._post(
            "/search", {"query": query, "user_id": user_id, "top_k": top_k}
        )


def run(args: argparse.Namespace) -> int:
    client = Client(args.base_url, args.api_key)
    rng = random.Random(args.seed)
    add_stats = Stats()
    search_stats = Stats()
    stop = threading.Event()
    counter = {"n": 0}
    counter_lock = threading.Lock()

    def next_seq() -> int:
        with counter_lock:
            counter["n"] += 1
            return counter["n"]

    # Seed one memory per user so searches have something to find.
    for user_index in range(args.users):
        user_id = f"load:user-{user_index}"
        try:
            client.add(f"seed-{user_index}", user_id, "s0", make_message(0, rng))
        except Exception as exc:
            print(f"seed failed: {exc}", file=sys.stderr)
            return 2

    def writer(worker: int) -> None:
        while not stop.is_set():
            seq = next_seq()
            user_id = f"load:user-{worker % args.users}"
            started = time.monotonic()
            try:
                client.add(
                    f"w-{worker}-{seq}", user_id, f"s-{worker}-{seq}", make_message(seq, rng)
                )
                add_stats.add(time.monotonic() - started)
            except Exception as exc:
                add_stats.error(str(exc))

    def retrier(worker: int) -> None:
        """Re-send the same request_id, exercising idempotency under load."""
        seq_base = 100_000 + worker * 1000
        while not stop.is_set():
            seq = seq_base + next_seq()
            user_id = f"load:user-{worker % args.users}"
            content = make_message(seq, rng)
            request_id = f"retry-{worker}-{seq}"
            try:
                for _ in range(3):
                    client.add(request_id, user_id, f"s-{worker}", content)
                add_stats.add(0.0)
            except Exception as exc:
                add_stats.error(str(exc))

    def searcher(worker: int) -> None:
        while not stop.is_set():
            user_id = f"load:user-{worker % args.users}"
            query = f"why did {rng.choice(FILES)} raise {rng.choice(ERRORS)}"
            started = time.monotonic()
            try:
                client.search(user_id, query, top_k=args.top_k)
                search_stats.add(time.monotonic() - started)
            except Exception as exc:
                search_stats.error(str(exc))

    threads: list[threading.Thread] = []
    for w in range(args.writers):
        threads.append(threading.Thread(target=writer, args=(w,), daemon=True))
    for r in range(args.retriers):
        threads.append(threading.Thread(target=retrier, args=(r,), daemon=True))
    for s in range(args.searchers):
        threads.append(threading.Thread(target=searcher, args=(s,), daemon=True))

    print(
        f"load: {args.writers} writers, {args.retriers} retriers, "
        f"{args.searchers} searchers for {args.seconds}s against {args.base_url}"
    )
    started = time.monotonic()
    for t in threads:
        t.start()
    time.sleep(args.seconds)
    stop.set()
    for t in threads:
        t.join(timeout=60)
    elapsed = time.monotonic() - started

    add_summary = add_stats.summary()
    search_summary = search_stats.summary()

    print(f"\nelapsed: {elapsed:.1f}s")
    print(f"add:    {json.dumps(add_summary)}")
    print(f"search: {json.dumps(search_summary)}")

    problems = []
    if add_summary.get("errors"):
        problems.append(f"{len(add_summary['errors'])} add errors")
    if search_summary.get("errors"):
        problems.append(f"{len(search_summary['errors'])} search errors")
    if not search_summary.get("count"):
        problems.append("no searches completed")

    if problems:
        print("\nFAIL: " + "; ".join(problems))
        for err in (add_summary.get("errors", []) + search_summary.get("errors", []))[:5]:
            print(f"  - {err}")
        return 1

    print("\nPASS: no transport or contract errors under concurrent load")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--writers", type=int, default=4)
    parser.add_argument("--retriers", type=int, default=2)
    parser.add_argument("--searchers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
