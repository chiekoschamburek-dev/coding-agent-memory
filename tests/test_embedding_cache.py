"""Cache durability, migration, concurrency and memory-budget regressions."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codemem.common.config import Settings
from codemem.common.embeddings import EmbeddingCache, get_embedder


def test_legacy_import_streams_without_prewarming_or_rewriting(tmp_path, monkeypatch):
    path = tmp_path / "vectors.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(1000):
            f.write(json.dumps({"k": str(i), "v": [i / 1000, 0.12345678901234567]}) + "\n")
    before = path.read_bytes()

    def no_bulk_read(*args, **kwargs):
        raise AssertionError("legacy migration must stream its input")

    monkeypatch.setattr(Path, "read_text", no_bulk_read)
    cache = EmbeddingCache(path, memory_max_bytes=4096)
    assert cache.get("999") == [0.999, 0.12345678901234567]
    stats = cache.stats()
    assert stats["entries"] == 1000
    assert stats["memory_entries"] == 1
    assert stats["memory_bytes"] <= 4096
    assert path.read_bytes() == before
    cache.close()


def test_legacy_import_skips_bad_records_and_keeps_last_duplicate(tmp_path):
    path = tmp_path / "vectors.jsonl"
    path.write_text('\nnot json\n[]\n{"k":"bad","v":"wrong"}\n'
                    '{"k":"good","v":[1.0]}\n{"k":"good","v":[2.0]}\n',
                    encoding="utf-8")
    cache = EmbeddingCache(path)
    assert cache.get("good") == [2.0]
    assert cache.stats()["entries"] == 1
    cache.close()


def test_completed_import_is_not_repeated(tmp_path):
    path = tmp_path / "vectors.jsonl"
    path.write_text('{"k":"old","v":[1.0]}\n', encoding="utf-8")
    cache = EmbeddingCache(path)
    assert cache.get("old") == [1.0]
    cache.put("new", [2.0])
    cache.close()
    path.write_text("not a valid legacy cache anymore", encoding="utf-8")
    reopened = EmbeddingCache(path)
    assert reopened.get("old") == [1.0]
    assert reopened.get("new") == [2.0]
    reopened.close()


def test_interrupted_import_rolls_back_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "vectors.jsonl"
    path.write_text('{"k":"a","v":[1.0]}\n{"k":"b","v":[2.0]}\n',
                    encoding="utf-8")
    cache = EmbeddingCache(path)
    original = cache._pack
    calls = 0

    def interrupted(vector):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted legacy read")
        return original(vector)

    monkeypatch.setattr(cache, "_pack", interrupted)
    assert cache.get("a") is None
    with sqlite3.connect(cache.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM cache_meta").fetchone()[0] == 0
    monkeypatch.setattr(cache, "_pack", original)
    assert cache.get("a") == [1.0]
    assert cache.get("b") == [2.0]
    cache.close()


def test_lru_evicts_by_bytes_and_recovers_evicted_vectors_from_disk(tmp_path):
    cache = EmbeddingCache(tmp_path / "vectors.jsonl", memory_max_bytes=19000)
    vector = [0.12345678901234567] * 1024
    cache.put("a", vector)
    cache.put("b", vector)
    assert cache.get("a") == vector  # a is now newer than b
    cache.put("c", vector)
    stats = cache.stats()
    assert stats["entries"] == 3
    assert stats["memory_entries"] == 2
    assert stats["memory_bytes"] <= stats["memory_max_bytes"]
    before = stats["disk_hits"]
    assert cache.get("a") == vector
    assert cache.stats()["disk_hits"] == before
    assert cache.get("b") == vector  # evicted, but not lost
    assert cache.stats()["disk_hits"] == before + 1
    assert cache.stats()["memory_bytes"] <= 19000
    cache.close()


def test_zero_budget_and_oversize_entries_stay_disk_only(tmp_path):
    for budget in [0, 128]:
        cache = EmbeddingCache(tmp_path / f"vectors-{budget}.jsonl", memory_max_bytes=budget)
        vector = [0.1] * 1024
        cache.put("large", vector)
        assert cache.get("large") == vector
        stats = cache.stats()
        assert stats["memory_entries"] == 0
        assert stats["memory_bytes"] == 0
        assert stats["disk_hits"] == 1
        cache.close()


def test_disk_writes_survive_reopen_with_exact_float_values(tmp_path):
    path = tmp_path / "vectors.jsonl"
    vector = [0.12345678901234567, -1e-100, 1e100]
    cache = EmbeddingCache(path, memory_max_bytes=0)
    cache.put("key", vector)
    cache.close()
    assert not path.exists()  # no duplicate JSONL storage for new writes
    reopened = EmbeddingCache(path)
    result = reopened.get("key")
    assert result == vector
    result[0] = 999  # callers cannot mutate the retained cached vector
    assert reopened.get("key") == vector
    reopened.put("key", [42.0])
    assert reopened.get("key") == vector
    reopened.close()


def test_concurrent_readers_and_writers_across_instances(tmp_path):
    path = tmp_path / "vectors.jsonl"
    caches = [EmbeddingCache(path, memory_max_bytes=4096) for _ in range(2)]

    def write_and_read(i):
        cache = caches[i % 2]
        vector = [i / 10, 0.12345678901234567]
        cache.put(str(i), vector)
        assert cache.get(str(i)) == vector

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_and_read, range(100)))
    for cache in caches:
        stats = cache.stats()
        assert stats["entries"] == 100
        assert stats["memory_bytes"] <= 4096
        assert stats["errors"] == 0
        cache.close()


def test_disk_failure_is_a_cache_miss_without_raising(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file", encoding="utf-8")
    cache = EmbeddingCache(blocked / "vectors.jsonl")
    assert cache.get("key") is None
    cache.put("key", [1.0])
    assert cache.stats()["errors"] >= 2


def test_factory_respects_configured_memory_budget(tmp_path):
    cfg = Settings(embed_cache_memory_mb=0)
    client = get_embedder(cfg, tmp_path / "vectors.jsonl")
    assert client.cache.memory_max_bytes == 0
    client.cache.close()
