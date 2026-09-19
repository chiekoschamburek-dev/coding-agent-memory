"""Concurrency.

The platform may retry Add (up to 32 attempts, same request_id) and can issue
concurrent Search calls during a Full run. WAL mode plus a serialized writer
must keep that correct: no lost writes, no duplicate memory, no reader errors.
"""

from __future__ import annotations

import concurrent.futures as futures

from codemem.core.config import Settings
from codemem.index.store import Store


def test_concurrent_adds_do_not_lose_writes(settings):
    store = Store(settings)
    try:
        from codemem.add.pipeline import AddPipeline

        pipeline = AddPipeline(settings, store)

        class M:
            def __init__(self, content):
                self.role = "user"
                self.content = content
                self.timestamp = None

        def write(i: int):
            return pipeline.handle(
                request_id=f"req-{i}",
                user_id="u1",
                session_id=f"s{i}",
                messages=[M(f"Session {i} touched src/mod{i}/file.py with unique marker {i}.")],
            )

        with futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(write, range(24)))

        assert all(not r[0].duplicate for r in results)
        assert store.counts()[1] == 24, "every distinct write must persist"
    finally:
        store.close()


def test_concurrent_retries_of_one_request_id_write_once(settings):
    store = Store(settings)
    try:
        from codemem.add.pipeline import AddPipeline

        pipeline = AddPipeline(settings, store)

        class M:
            role = "user"
            content = "A single fact about src/only/once.py that must appear once."
            timestamp = None

        def write(_):
            return pipeline.handle(
                request_id="same-request",
                user_id="u1",
                session_id="s1",
                messages=[M()],
            )

        with futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(16)))

        assert store.counts()[1] == 1, "concurrent retries must not duplicate"
        row = store.get_request("same-request")
        assert row is not None
    finally:
        store.close()


def test_interleaved_users_stay_separate(settings):
    store = Store(settings)
    try:
        from codemem.add.pipeline import AddPipeline

        pipeline = AddPipeline(settings, store)

        class M:
            def __init__(self, content):
                self.role = "user"
                self.content = content
                self.timestamp = None

        def write(i: int):
            user = f"user-{i % 4}"
            return pipeline.handle(
                request_id=f"req-{i}",
                user_id=user,
                session_id=f"s{i}",
                messages=[M(f"Marker {i} belongs to {user} in src/{user}/mod.py.")],
            )

        with futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(32)))

        # Each user must see exactly its own 8 memories and no other's.
        for u in range(4):
            user = f"user-{u}"
            hits = store.bm25_search(user, f"belongs to {user}", 100)
            assert len(hits) == 8, f"{user} saw {len(hits)} memories"
    finally:
        store.close()


def test_concurrent_search_during_writes(settings):
    store = Store(settings)
    try:
        from codemem.add.pipeline import AddPipeline
        from codemem.search.service import SearchPipeline

        add = AddPipeline(settings, store)
        search = SearchPipeline(settings, store)

        class M:
            def __init__(self, content):
                self.role = "user"
                self.content = content
                self.timestamp = None

        add.handle(
            request_id="seed",
            user_id="u1",
            session_id="s0",
            messages=[M("Seed memory about src/seed/module.py and its retry policy.")],
        )

        def busy(i: int):
            if i % 2 == 0:
                add.handle(
                    request_id=f"w{i}",
                    user_id="u1",
                    session_id=f"s{i}",
                    messages=[M(f"Write {i} about src/w{i}/mod.py with token{i}.")],
                )
                return "write"
            search.handle(
                user_id="u1", query="retry policy module", options=None, top_k=50
            )
            return "read"

        with futures.ThreadPoolExecutor(max_workers=8) as pool:
            kinds = list(pool.map(busy, range(40)))
        assert kinds.count("write") == 20
        assert kinds.count("read") == 20
    finally:
        store.close()
