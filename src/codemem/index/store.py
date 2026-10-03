"""SQLite-backed store.

Concurrency model
-----------------
SQLite in WAL mode allows many concurrent readers with a single writer. The
platform issues Add retries and may interleave calls, so:

* a single connection guarded by a lock handles writes (``busy_timeout`` also
  set, for safety against external readers);
* reads use a small pool of connections, each in WAL read mode.

Every public method takes ``user_id`` and filters on it. There is no method
that reads memory without a ``user_id`` — isolation is enforced by the API
shape, not by remembering to add a WHERE clause.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..core.config import Settings
from ..core.logging import get_logger
from ..embed import pack as _pack
from ..embed import unpack as _unpack
from .schema import DDL
from .sparse import build_sparse

log = get_logger("codemem.index")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass(slots=True)
class MemoryRow:
    id: int
    user_id: str
    session_id: str
    request_id: str
    chunk_id: int | None
    kind: str
    title: str | None
    text: str
    ts: int | None
    ord: int
    created_at: str
    superseded_by: int | None = None

    @property
    def structural_kind(self) -> str:
        """The chunker's kind, carried on the title as ``kind|lang``."""
        if self.title and "|" in self.title:
            return self.title.split("|", 1)[0]
        return self.kind

    @property
    def structural_lang(self) -> str | None:
        if self.title and "|" in self.title:
            lang = self.title.split("|", 1)[1]
            return lang or None
        return None


@dataclass(slots=True)
class ChunkRow:
    id: int
    user_id: str
    session_id: str
    request_id: str
    msg_index: int
    part_index: int
    part_count: int
    kind: str
    lang: str | None
    text: str
    sha: str
    line_start: int
    line_end: int
    ts: int | None
    ord: int
    created_at: str


@dataclass(slots=True)
class AddOutcome:
    memories_written: int
    chunks_written: int
    duplicate: bool


class Store:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.db_path: Path = settings.db_path
        self._write_lock = threading.RLock()
        self._pool_lock = threading.Lock()
        self._read_conns: list[sqlite3.Connection] = []
        self._max_read_conns = 8
        self._closed = False
        self._init_db()

    # ------------------------------------------------------------ setup --

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # explicit transactions
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA cache_size=-64000")
        return conn

    def _init_db(self) -> None:
        with self._write_lock:
            conn = self._connect()
            try:
                conn.executescript(DDL)
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    ("1",),
                )
            finally:
                conn.close()
        log.info("store initialised", extra={"ctx": {"db": str(self.db_path)}})

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._acquire_read()
        try:
            yield conn
        finally:
            self._release_read(conn)

    def _acquire_read(self) -> sqlite3.Connection:
        with self._pool_lock:
            if self._read_conns:
                return self._read_conns.pop()
        return self._connect()

    def _release_read(self, conn: sqlite3.Connection) -> None:
        with self._pool_lock:
            if not self._closed and len(self._read_conns) < self._max_read_conns:
                self._read_conns.append(conn)
                return
        conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover - rollback best effort
                    pass
                raise
            finally:
                conn.close()

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            conns, self._read_conns = self._read_conns, []
        for conn in conns:
            conn.close()

    # ------------------------------------------------------------- idem --

    def get_request(self, request_id: str) -> sqlite3.Row | None:
        with self._read() as conn:
            return conn.execute(
                "SELECT request_id, user_id, session_id, n_messages, n_memories, created_at "
                "FROM request_seen WHERE request_id = ?",
                (request_id,),
            ).fetchone()

    # ----------------------------------------------------------- writes --

    def add_messages(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: Sequence[dict[str, Any]],
        chunks: Sequence[dict[str, Any]],
        entities: dict[int, list[tuple[str, str, str]]],
        t0: float,
    ) -> AddOutcome:
        """Persist raw messages + chunks + entities in one transaction.

        The transaction commits only after the FTS index is updated by trigger,
        so once this returns the memory is durable and searchable — the
        precondition for answering ``success: true``.

        Returns ``duplicate=True`` when this ``request_id`` was already stored,
        in which case nothing is written.
        """
        now = utc_now_iso()
        with self._write() as conn:
            existing = conn.execute(
                "SELECT n_memories FROM request_seen WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing is not None:
                return AddOutcome(0, 0, duplicate=True)

            for msg in messages:
                conn.execute(
                    "INSERT OR IGNORE INTO raw_message"
                    "(user_id, session_id, request_id, msg_index, role, ts, content,"
                    " content_sha, chars, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        user_id,
                        session_id,
                        request_id,
                        msg["msg_index"],
                        msg["role"],
                        msg.get("ts"),
                        msg["content"],
                        sha256_text(msg["content"]),
                        len(msg["content"]),
                        now,
                    ),
                )

            written = 0
            for chunk in chunks:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO chunk"
                    "(user_id, session_id, request_id, msg_index, part_index, part_count,"
                    " kind, lang, text, sha, line_start, line_end, ts, ord, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        user_id,
                        session_id,
                        request_id,
                        chunk["msg_index"],
                        chunk["part_index"],
                        chunk["part_count"],
                        chunk["kind"],
                        chunk.get("lang"),
                        chunk["text"],
                        chunk["sha"],
                        chunk["line_start"],
                        chunk["line_end"],
                        chunk.get("ts"),
                        chunk["ord"],
                        now,
                    ),
                )
                if cur.rowcount == 0:
                    # Same (user, sha, request) already present: reuse its id so
                    # nothing is duplicated within one logical write.
                    row = conn.execute(
                        "SELECT id FROM chunk WHERE user_id=? AND sha=? AND request_id=?",
                        (user_id, chunk["sha"], request_id),
                    ).fetchone()
                    chunk_id = int(row["id"]) if row else None
                    if chunk_id is None:
                        continue
                else:
                    chunk_id = int(cur.lastrowid or 0)

                for etype, value_norm, value_raw in entities.get(chunk["ord"], []):
                    conn.execute(
                        "INSERT OR IGNORE INTO chunk_entity"
                        "(chunk_id, user_id, etype, value_norm, value_raw) VALUES (?,?,?,?,?)",
                        (chunk_id, user_id, etype, value_norm, value_raw),
                    )

                memory_id = self._insert_memory(
                    conn,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    chunk_id=chunk_id,
                    kind=chunk.get("memory_kind", "chunk"),
                    title=chunk.get("title"),
                    text=chunk["text"],
                    ts=chunk.get("ts"),
                    ord=chunk["ord"],
                    now=now,
                    entities=entities.get(chunk["ord"], []),
                )
                if memory_id is None:
                    continue
                written += 1

            conn.execute(
                "INSERT INTO request_seen"
                "(request_id, user_id, session_id, n_messages, n_memories, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (request_id, user_id, session_id, len(messages), written, now),
            )
            self._refresh_repo_profile(conn, user_id, now)

        elapsed = time.monotonic() - t0
        log.info(
            "add committed",
            extra={
                "ctx": {
                    "request_id": request_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": len(messages),
                    "chunks": len(chunks),
                    "memories": written,
                    "elapsed_s": round(elapsed, 3),
                }
            },
        )
        return AddOutcome(written, len(chunks), duplicate=False)

    def _insert_memory(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        chunk_id: int | None,
        kind: str,
        title: str | None,
        text: str,
        ts: int | None,
        ord: int,
        now: str,
        entities: Sequence[tuple[str, str, str]] = (),
    ) -> int | None:
        sha = sha256_text(text)
        sparse = build_sparse(text, title)
        cur = conn.execute(
            "INSERT OR IGNORE INTO memory"
            "(user_id, session_id, request_id, chunk_id, kind, title, text, sparse,"
            " sha, ts, ord, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                user_id, session_id, request_id, chunk_id, kind, title, text, sparse,
                sha, ts, ord, now,
            ),
        )
        if cur.rowcount == 0:
            return None
        memory_id = int(cur.lastrowid or 0)
        for etype, value_norm, value_raw in entities:
            conn.execute(
                "INSERT OR IGNORE INTO entity_timeline"
                "(user_id, etype, value_norm, memory_id, ts, weight) VALUES (?,?,?,?,?,?)",
                (user_id, etype, value_norm, memory_id, ts, 1.0),
            )
        return memory_id

    def _refresh_repo_profile(self, conn: sqlite3.Connection, user_id: str, now: str) -> None:
        """Rebuild the entity document-frequency table for a user_id.

        Two uses:

        * **Soft filter / diagnostic** — the dominant directories and languages
          characterise the repository. A bimodal profile means one ``user_id``
          actually spans several repositories, in which case the profile should
          partition candidates rather than be trusted blindly.
        * **IDF for identifier weighting** — an identifier appearing in one
          chunk is highly discriminative; one appearing in hundreds is nearly
          useless. Under same-repository noise this is what separates a real
          clue from a shared file path.
        """
        conn.execute("DELETE FROM repo_profile WHERE user_id = ?", (user_id,))
        conn.execute(
            "INSERT INTO repo_profile(user_id, etype, value_norm, n, updated_at)"
            " SELECT user_id, etype, value_norm, count(DISTINCT chunk_id), ?"
            " FROM chunk_entity WHERE user_id = ?"
            " GROUP BY user_id, etype, value_norm",
            (now, user_id),
        )

    # ------------------------------------------------------------ reads --

    def counts(self) -> tuple[int, int]:
        with self._read() as conn:
            users = conn.execute("SELECT count(DISTINCT user_id) c FROM memory").fetchone()["c"]
            memories = conn.execute("SELECT count(*) c FROM memory").fetchone()["c"]
        return int(users), int(memories)

    def user_has_memory(self, user_id: str) -> bool:
        with self._read() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory WHERE user_id = ? LIMIT 1", (user_id,)
            ).fetchone()
        return row is not None

    def repo_profile(self, user_id: str) -> dict[str, list[tuple[str, int]]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT etype, value_norm, n FROM repo_profile WHERE user_id = ?"
                " ORDER BY n DESC",
                (user_id,),
            ).fetchall()
        out: dict[str, list[tuple[str, int]]] = {}
        for row in rows:
            out.setdefault(row["etype"], []).append((row["value_norm"], int(row["n"])))
        return out

    def bm25_search(self, user_id: str, query: str, limit: int) -> list[tuple[int, float]]:
        """FTS5 BM25 over one user's memory. Returns (memory_id, score)."""
        from .sparse import escape_fts_token, fts_query_terms

        terms = fts_query_terms(query)
        if not terms:
            return []
        expr = " OR ".join(escape_fts_token(t) for t in terms)
        try:
            with self._read() as conn:
                rows = conn.execute(
                    "SELECT f.rowid AS rid, bm25(memory_fts, 1.0, 1.5, 0.8) AS b "
                    "FROM memory_fts f "
                    "JOIN memory m ON m.id = f.rowid "
                    "WHERE memory_fts MATCH ? AND m.user_id = ? "
                    "ORDER BY b LIMIT ?",
                    (expr, user_id, limit),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning(
                "bm25 query failed",
                extra={"ctx": {"user_id": user_id, "error": str(exc)}},
            )
            return []
        # bm25() returns a negative value (more negative = better).
        return [(int(r["rid"]), -float(r["b"])) for r in rows]

    def entity_match_scores(
        self, user_id: str, entity_hits: dict[str, list[str]]
    ) -> dict[int, float]:
        """Sum IDF-weighted identifier matches, keyed by ``chunk_id``.

        Single source of truth for identifier evidence, shared by the recall
        channel and the scorer so the two can never disagree. Exact equality
        only — no fuzzy matching — which is precisely why this channel holds up
        when the memory pool is full of same-repository distractors.
        """
        from ..add.entities import ENTITY_WEIGHT

        clauses: list[str] = []
        params: list[Any] = [user_id]
        for etype, values in entity_hits.items():
            if not values:
                continue
            placeholders = ",".join("?" for _ in values)
            clauses.append(f"(ce.etype = ? AND ce.value_norm IN ({placeholders}))")
            params.append(etype)
            params.extend(values)
        if not clauses:
            return {}

        sql = (
            "SELECT ce.chunk_id AS chunk_id, ce.etype AS etype,"
            " ce.value_norm AS value_norm"
            " FROM chunk_entity ce WHERE ce.user_id = ? AND ("
            + " OR ".join(clauses)
            + ")"
        )
        with self._read() as conn:
            df_rows = conn.execute(
                "SELECT etype, value_norm, n FROM repo_profile WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            df = {(r["etype"], r["value_norm"]): int(r["n"]) for r in df_rows}
            row = conn.execute(
                "SELECT count(*) c FROM chunk WHERE user_id = ?", (user_id,)
            ).fetchone()
            total = max(1, int(row["c"] if row else 1))
            rows = conn.execute(sql, params).fetchall()

        scores: dict[int, float] = {}
        seen: set[tuple[int, str, str]] = set()
        for r in rows:
            key = (int(r["chunk_id"]), r["etype"], r["value_norm"])
            if key in seen:
                continue
            seen.add(key)
            base = ENTITY_WEIGHT.get(r["etype"], 0.2)
            n = df.get((r["etype"], r["value_norm"]), 1)
            idf = 1.0 + math.log(total / max(1, n))
            chunk_id = key[0]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + base * idf
        return scores

    def entity_search(
        self, user_id: str, entity_hits: dict[str, list[str]], limit: int
    ) -> list[tuple[int, float]]:
        """Recall channel: chunks with matching identifiers, best first."""
        chunk_scores = self.entity_match_scores(user_id, entity_hits)
        if not chunk_scores:
            return []
        top_chunks = sorted(chunk_scores.items(), key=lambda kv: -kv[1])[: max(limit, 1)]
        chunk_ids = [cid for cid, _ in top_chunks]
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._read() as conn:
            mem = conn.execute(
                "SELECT id, chunk_id FROM memory"
                f" WHERE user_id = ? AND chunk_id IN ({placeholders})",
                [user_id, *chunk_ids],
            ).fetchall()
        # A memory's score is the best entity score among the chunks it covers.
        best: dict[int, float] = {}
        for row in mem:
            score = chunk_scores.get(int(row["chunk_id"]), 0.0)
            memory_id = int(row["id"])
            if score > best.get(memory_id, 0.0):
                best[memory_id] = score
        return sorted(best.items(), key=lambda kv: -kv[1])[:limit]

    def session_map(self, user_id: str, memory_ids: Sequence[int]) -> dict[int, str]:
        """Batch memory_id -> session_id, for session-major candidate assembly.

        One query per call rather than one per memory: the candidate pool is a
        few hundred ids, and a per-id lookup would add that many round trips to
        every search.
        """
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT id, session_id FROM memory WHERE user_id = ?"
                f" AND id IN ({placeholders})",
                (user_id, *memory_ids),
            ).fetchall()
        return {int(row["id"]): row["session_id"] for row in rows}

    def session_span(
        self, user_id: str, session_ids: Sequence[str]
    ) -> dict[str, tuple[int, int]]:
        """Lowest and highest memory row id per session.

        Row ids are assigned in insertion order, which for a trajectory is the
        order the platform sent the messages in - including across several Add
        calls for one session, where ``ord`` restarts per request. Assembly uses
        it to place a candidate inside its own session (0 = first message,
        1 = last), which is the only position scale that means the same thing for
        a 20-turn and a 300-turn trajectory.
        """
        ids = [s for s in session_ids if s]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT session_id, min(id) AS lo, max(id) AS hi FROM memory"
                f" WHERE user_id = ? AND session_id IN ({placeholders})"
                " GROUP BY session_id",
                (user_id, *ids),
            ).fetchall()
        return {
            row["session_id"]: (int(row["lo"]), int(row["hi"])) for row in rows
        }

    def fetch_memories(self, user_id: str, memory_ids: Sequence[int]) -> dict[int, MemoryRow]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT id, user_id, session_id, request_id, chunk_id, kind, title, text,"
                " ts, ord, created_at, superseded_by FROM memory"
                f" WHERE user_id = ? AND id IN ({placeholders})",
                [user_id, *memory_ids],
            ).fetchall()
        return {
            int(r["id"]): MemoryRow(
                id=int(r["id"]),
                user_id=r["user_id"],
                session_id=r["session_id"],
                request_id=r["request_id"],
                chunk_id=int(r["chunk_id"]) if r["chunk_id"] is not None else None,
                kind=r["kind"],
                title=r["title"],
                text=r["text"],
                ts=int(r["ts"]) if r["ts"] is not None else None,
                ord=int(r["ord"]),
                created_at=r["created_at"],
                superseded_by=(
                    int(r["superseded_by"]) if r["superseded_by"] is not None else None
                ),
            )
            for r in rows
        }

    def fetch_chunks(self, user_id: str, chunk_ids: Sequence[int]) -> dict[int, ChunkRow]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT id, user_id, session_id, request_id, msg_index, part_index, part_count,"
                " kind, lang, text, sha, line_start, line_end, ts, ord, created_at FROM chunk"
                f" WHERE user_id = ? AND id IN ({placeholders})",
                [user_id, *chunk_ids],
            ).fetchall()
        return {
            int(r["id"]): ChunkRow(
                id=int(r["id"]),
                user_id=r["user_id"],
                session_id=r["session_id"],
                request_id=r["request_id"],
                msg_index=int(r["msg_index"]),
                part_index=int(r["part_index"]),
                part_count=int(r["part_count"]),
                kind=r["kind"],
                lang=r["lang"],
                text=r["text"],
                sha=r["sha"],
                line_start=int(r["line_start"]),
                line_end=int(r["line_end"]),
                ts=int(r["ts"]) if r["ts"] is not None else None,
                ord=int(r["ord"]),
                created_at=r["created_at"],
            )
            for r in rows
        }

    def session_recency(self, user_id: str, session_ids: Sequence[str]) -> dict[str, int]:
        """Latest timestamp seen per session, for recency weighting."""
        if not session_ids:
            return {}
        placeholders = ",".join("?" for _ in session_ids)
        with self._read() as conn:
            rows = conn.execute(
                "SELECT session_id, max(coalesce(ts, 0)) AS m FROM chunk"
                f" WHERE user_id = ? AND session_id IN ({placeholders}) GROUP BY session_id",
                [user_id, *session_ids],
            ).fetchall()
        return {r["session_id"]: int(r["m"]) for r in rows}

    def all_entity_values(self, user_id: str, etype: str, limit: int = 500) -> list[str]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT value_norm FROM repo_profile WHERE user_id=? AND etype=?"
                " ORDER BY n DESC LIMIT ?",
                (user_id, etype, limit),
            ).fetchall()
        return [r["value_norm"] for r in rows]

    def superseded_map(self, user_id: str) -> dict[int, int]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT s.old_memory_id, s.new_memory_id FROM supersedes s"
                " JOIN memory m ON m.id = s.old_memory_id WHERE m.user_id = ?",
                (user_id,),
            ).fetchall()
        return {int(r["old_memory_id"]): int(r["new_memory_id"]) for r in rows}

    def max_ts(self, user_id: str) -> int:
        with self._read() as conn:
            row = conn.execute(
                "SELECT max(coalesce(ts,0)) m FROM memory WHERE user_id = ?", (user_id,)
            ).fetchone()
        return int(row["m"] or 0)

    # ----------------------------------------------------------- delete --

    def store_vectors(
        self, user_id: str, items: Sequence[tuple[int, Sequence[float]]]
    ) -> int:
        """Persist embeddings for memories, keyed by user.

        ``INSERT OR REPLACE`` keeps re-embedding idempotent, which matters
        because enrichment may be retried.
        """
        if not items:
            return 0
        written = 0
        with self._write() as conn:
            for memory_id, vector in items:
                if not vector:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO memory_vector(memory_id, user_id, dim, vec)"
                    " VALUES (?,?,?,?)",
                    (memory_id, user_id, len(vector), _pack(vector)),
                )
                written += 1
        return written

    def dense_search(
        self, user_id: str, query_vector: Sequence[float], limit: int
    ) -> list[tuple[int, float]]:
        """Exact nearest neighbours within one user, by cosine similarity.

        Brute force over a user's own vectors on purpose: the platform's top_k is
        100 and a single user's corpus is bounded by its trajectories, so an
        approximate index would trade correctness for a speedup we do not need —
        and every ANN structure would add a second place where isolation could be
        got wrong. Reads only this user's rows.
        """
        if not query_vector:
            return []
        with self._read() as conn:
            rows = conn.execute(
                "SELECT memory_id, vec FROM memory_vector WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        if not rows:
            return []

        q = list(query_vector)
        scored: list[tuple[int, float]] = []
        for row in rows:
            vec = _unpack(row["vec"])
            if len(vec) != len(q):
                continue  # dimension drift from a changed encoder; skip
            # Vectors are stored normalized, so the dot product is cosine.
            score = 0.0
            for a, b in zip(q, vec):
                score += a * b
            scored.append((int(row["memory_id"]), score))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:limit]

    def vector_coverage(self, user_id: str) -> tuple[int, int]:
        """(embedded, total) memories for a user, to detect a stale index."""
        with self._read() as conn:
            total = conn.execute(
                "SELECT count(*) c FROM memory WHERE user_id = ?", (user_id,)
            ).fetchone()["c"]
            embedded = conn.execute(
                "SELECT count(*) c FROM memory_vector WHERE user_id = ?", (user_id,)
            ).fetchone()["c"]
        return int(embedded), int(total)

    def mark_superseded(self, user_id: str, old_memory_id: int, new_memory_id: int, reason: str) -> None:
        """Record that a newer memory supersedes an older one.

        The scoring path already applies a 0.65 penalty to superseded memories.
        Nothing calls this yet: deciding when a later session genuinely
        invalidates an earlier one needs calibration against the benchmark, and
        an uncalibrated heuristic would silently down-weight valid evidence.
        The mechanism is in place so it can be switched on once
        ``eval/build_benchmark.py`` can measure the change.

        TODO(P3): populate from the enrichment pass, behind a config flag.
        """
        with self._write() as conn:
            owns_old = conn.execute(
                "SELECT 1 FROM memory WHERE id = ? AND user_id = ?",
                (old_memory_id, user_id),
            ).fetchone()
            owns_new = conn.execute(
                "SELECT 1 FROM memory WHERE id = ? AND user_id = ?",
                (new_memory_id, user_id),
            ).fetchone()
            if not (owns_old and owns_new):
                raise ValueError("both memories must belong to this user_id")
            conn.execute(
                "INSERT OR IGNORE INTO supersedes"
                "(old_memory_id, new_memory_id, reason, created_at) VALUES (?,?,?,?)",
                (old_memory_id, new_memory_id, reason, utc_now_iso()),
            )
            conn.execute(
                "UPDATE memory SET superseded_by = ? WHERE id = ? AND user_id = ?",
                (new_memory_id, old_memory_id, user_id),
            )

    def delete_user(self, user_id: str) -> dict[str, int]:
        """Hard-delete every trace of a user_id.

        Required by the retention obligation: evaluation data and its derived
        copies must be removed within 30 days of task completion.
        """
        removed: dict[str, int] = {}
        with self._write() as conn:
            memory_ids = [
                int(r["id"])
                for r in conn.execute(
                    "SELECT id FROM memory WHERE user_id = ?", (user_id,)
                ).fetchall()
            ]
            if memory_ids:
                placeholders = ",".join("?" for _ in memory_ids)
                conn.execute(
                    f"DELETE FROM supersedes WHERE old_memory_id IN ({placeholders})"
                    f" OR new_memory_id IN ({placeholders})",
                    [*memory_ids, *memory_ids],
                )
            for table in (
                "memory_vector",
                "entity_timeline",
                "memory",
                "chunk_entity",
                "chunk",
                "raw_message",
                "request_seen",
                "repo_profile",
            ):
                cur = conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                removed[table] = cur.rowcount
            conn.execute(
                "INSERT INTO memory_fts(memory_fts) VALUES('optimize')"
            )
        log.info("user deleted", extra={"ctx": {"user_id": user_id, "removed": removed}})
        return removed

    def vacuum(self) -> None:
        with self._write() as conn:
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('optimize')")
        conn = self._connect()
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
