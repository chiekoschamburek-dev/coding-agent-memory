"""Disk-backed embedding cache with a byte-bounded, thread-safe LRU.

The configured JSONL path is a legacy import source. New entries live in its
SQLite companion; the original file is never rewritten or loaded wholesale.
Vectors use float64 blobs to preserve the provider's Python float values.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import sys
import threading
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)
_EMPTY_LRU_BYTES = sys.getsizeof(OrderedDict())
_CACHE_ERRORS = (OSError, sqlite3.Error, ValueError, TypeError, struct.error, OverflowError)


class EmbeddingCache:
    def __init__(self, path: Path, *, memory_max_bytes: int = 16 * 1024 * 1024) -> None:
        self.path = Path(path)
        self.db_path = self.path.with_name(self.path.name + ".sqlite3")
        self.memory_max_bytes = max(0, memory_max_bytes)
        self._memory: OrderedDict[str, bytes] = OrderedDict()
        self._payload_bytes = 0
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self.hits = self.misses = self.memory_hits = self.disk_hits = self.errors = 0
        self._warned = False

    @staticmethod
    def _pack(vector: list[float]) -> bytes:
        if not isinstance(vector, list) or not vector:
            raise ValueError("cache vector must be a non-empty list")
        return struct.pack(f"<{len(vector)}d", *vector)

    @staticmethod
    def _unpack(blob: bytes) -> list[float]:
        if not blob or len(blob) % 8:
            raise ValueError("invalid cached vector size")
        return list(struct.unpack(f"<{len(blob) // 8}d", blob))

    def _database(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5, check_same_thread=False,
                               isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-2048")
            conn.execute("PRAGMA mmap_size=0")
            conn.execute("CREATE TABLE IF NOT EXISTS vectors "
                         "(key TEXT PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID")
            conn.execute("CREATE TABLE IF NOT EXISTS cache_meta "
                         "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
            # The transaction makes concurrent initialisers and interrupted
            # imports safe. A failed import rolls back and is retried later.
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM cache_meta WHERE key='jsonl_imported'").fetchone() is None:
                if self.path.exists():
                    with self.path.open(encoding="utf-8") as source:
                        for line in source:
                            try:
                                row = json.loads(line)
                                key = row["k"]
                                if not isinstance(key, str):
                                    continue
                                blob = self._pack(row["v"])
                            except (ValueError, TypeError, KeyError, struct.error, OverflowError):
                                continue
                            # Match the legacy reader's last-record-wins rule.
                            conn.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                         (key, blob))
                conn.execute("INSERT INTO cache_meta VALUES ('jsonl_imported', '1')")
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()
            raise
        self._conn = conn
        return conn

    def _memory_bytes(self) -> int:
        return self._payload_bytes + sys.getsizeof(self._memory) - _EMPTY_LRU_BYTES

    def _remember(self, key: str, blob: bytes) -> None:
        if self.memory_max_bytes == 0:
            return
        if key in self._memory:
            self._memory.move_to_end(key)
            return
        cost = sys.getsizeof(key) + sys.getsizeof(blob)
        # Oversize entries remain available on disk without displacing the LRU.
        if cost + 256 > self.memory_max_bytes:
            return
        self._memory[key] = blob
        self._payload_bytes += cost
        while self._memory and self._memory_bytes() > self.memory_max_bytes:
            old_key, old_blob = self._memory.popitem(last=False)
            self._payload_bytes -= sys.getsizeof(old_key) + sys.getsizeof(old_blob)
        if not self._memory:
            self._memory = OrderedDict()

    def _error(self, exc: Exception) -> None:
        self.errors += 1
        if not self._warned:
            # Never log provider credentials, cache keys, or source text.
            logger.warning("embedding cache unavailable (%s); continuing without it",
                           type(exc).__name__)
            self._warned = True

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            try:
                blob = self._memory.get(key)
                if blob is not None:
                    self._memory.move_to_end(key)
                    self.memory_hits += 1
                else:
                    row = self._database().execute(
                        "SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
                    if row is None:
                        self.misses += 1
                        return None
                    blob = row[0]
                    # Validate before adding an on-disk value to memory.
                    vector = self._unpack(blob)
                    self._remember(key, blob)
                    self.disk_hits += 1
                    self.hits += 1
                    return vector
                vector = self._unpack(blob)
                self.hits += 1
                return vector
            except _CACHE_ERRORS as exc:
                self._error(exc)
                self.misses += 1
                return None

    def put(self, key: str, vector: list[float]) -> None:
        with self._lock:
            try:
                blob = self._pack(vector)
                conn = self._database()
                inserted = conn.execute("INSERT OR IGNORE INTO vectors VALUES (?, ?)",
                                        (key, blob)).rowcount
                if not inserted:
                    # Preserve the first committed value, including when another
                    # cache instance already inserted this content key.
                    blob = conn.execute("SELECT vector FROM vectors WHERE key=?",
                                        (key,)).fetchone()[0]
                self._remember(key, blob)
            except _CACHE_ERRORS as exc:
                self._error(exc)

    def stats(self) -> dict:
        with self._lock:
            try:
                entries = self._database().execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            except _CACHE_ERRORS as exc:
                self._error(exc)
                entries = 0
            return {"hits": self.hits, "misses": self.misses, "entries": entries,
                    "memory_hits": self.memory_hits, "disk_hits": self.disk_hits,
                    "memory_entries": len(self._memory), "memory_bytes": self._memory_bytes(),
                    "memory_max_bytes": self.memory_max_bytes, "errors": self.errors}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._memory = OrderedDict()
            self._payload_bytes = 0
