"""Persistence for the card-only memory library.

Storage form (v6):

* The chunk is the persisted minimal fact. Raw messages are **not** stored;
  each Add commits its chunks (lossless segments of the messages) plus one
  ingestion row and one session row. Chunks are never searched directly --
  they exist as the evidence layer cards cite.
* Cards are extracted per *session*, after the session is fully added (a new
  Add with a different session_id finalises the previous one; Search
  finalises whatever is still pending as a backstop). A finalisation replaces
  the session's previous cards wholesale, so ids stay stable and extraction
  is idempotent.
* Only cards are retrieval-indexed: field BM25, entities, case
  anchors and card vectors. There is no chunk lexical index and no chunk
  vector store.

Every table is scoped by ``user_id`` and every query method takes ``user_id``
as its first argument so that isolation cannot be forgotten at a call site.
The lexical index is a hand-rolled inverted index + BM25 rather than SQLite
FTS5: code retrieval needs camelCase / snake_case splitting and verbatim
path matching, and one shared tokenizer between indexing and querying is
worth more here than FTS5's raw speed on a corpus of this size.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .config import Settings, get_settings

SCHEMA_VERSION = 9

# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
# A path-ish run must contain a separator or an extension dot to qualify, so
# ordinary sentences do not flood the index with junk tokens.
_PATHISH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\\-]{2,}[A-Za-z0-9_]")
_PATH_MARKER_RE = re.compile(r"[/\\]|\.[A-Za-z]{1,6}$")

_MAX_TOKEN_CHARS = 64


def _latin_tokens(text: str) -> Iterator[str]:
    for match in _WORD_RE.finditer(text):
        raw = match.group(0)
        low = raw.lower()
        if len(low) <= _MAX_TOKEN_CHARS:
            yield low
        # Split identifiers so `parseConfigFile` also matches `config`, `file`.
        if "_" in raw or (raw.upper() != raw and raw.lower() != raw):
            for part in re.split(r"_+", raw):
                if not part:
                    continue
                for sub in _CAMEL_RE.findall(part):
                    sub_low = sub.lower()
                    if 1 < len(sub_low) <= _MAX_TOKEN_CHARS and sub_low != low:
                        yield sub_low


def _cjk_tokens(text: str) -> Iterator[str]:
    """Character bigrams for CJK runs.

    History text and prompts may be Chinese; bigrams give usable recall
    without pulling in a segmentation dependency, and they degrade to single
    characters for isolated ideographs.
    """
    buf: list[str] = []

    def flush() -> Iterator[str]:
        if not buf:
            return
        if len(buf) == 1:
            yield buf[0]
        else:
            for i in range(len(buf) - 1):
                yield buf[i] + buf[i + 1]
        buf.clear()

    for ch in text:
        if "一" <= ch <= "鿿":
            buf.append(ch)
        else:
            yield from flush()
    yield from flush()


def tokenize(text: str) -> list[str]:
    """Single tokenizer shared by indexing and querying."""
    if not text:
        return []
    tokens = list(_latin_tokens(text))
    tokens.extend(_cjk_tokens(text))
    for match in _PATHISH_RE.finditer(text):
        run = match.group(0)
        if _PATH_MARKER_RE.search(run):
            low = run.lower().strip("._-/\\")
            if 1 < len(low) <= _MAX_TOKEN_CHARS:
                tokens.append(low)
            # Also index the basename so `token.py` matches `src/auth/token.py`.
            base = re.split(r"[/\\]", low)[-1]
            if base != low and 1 < len(base) <= _MAX_TOKEN_CHARS:
                tokens.append(base)
    return tokens


def term_frequencies(tokens: Sequence[str]) -> dict[str, int]:
    tf: dict[str, int] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return tf


# --------------------------------------------------------------------------
# ids
# --------------------------------------------------------------------------


def _digest(*parts: object) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


def content_hash(messages: Sequence["MessageInput"]) -> str:
    """Hash of the normalised request body.

    Order and exact string content both matter, so they are hashed directly
    rather than through any canonicalising pass.
    """
    hasher = hashlib.sha256()
    for message in messages:
        hasher.update(message.role.encode("utf-8"))
        hasher.update(b"\x1e")
        hasher.update(message.content.encode("utf-8"))
        hasher.update(b"\x1e")
        hasher.update(b"" if message.timestamp is None else str(message.timestamp).encode("ascii"))
        hasher.update(b"\x1d")
    return hasher.hexdigest()


def message_id_for(user_id: str, request_id: str, index: int) -> str:
    return "m:" + _digest(user_id, request_id, index)[:24]


def chunk_id_for(user_id: str, request_id: str, index: int, ord_: int, start: int, end: int) -> str:
    return "c:" + _digest(user_id, request_id, index, ord_, start, end)[:24]


def experience_id_for(user_id: str, request_id: str, local_id: int) -> str:
    return "e:" + _digest(user_id, request_id, local_id)[:24]


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MessageInput:
    role: str
    content: str
    timestamp: int | None = None


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    message_id: str
    message_index: int
    ord: int
    char_start: int
    char_end: int
    kind: str
    text: str
    parent_chunk_id: str | None
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class EntityRecord:
    chunk_id: str
    kind: str
    value: str


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    user_id: str
    message_id: str
    ord: int
    char_start: int
    char_end: int
    kind: str
    text: str
    source_ts: int | None
    ts_origin: str
    session_id: str
    role: str


@dataclass(frozen=True)
class IngestionState:
    ingestion_id: int
    user_id: str
    request_id: str
    session_id: str
    content_hash: str
    status: str
    visibility: str
    mode: str


class IngestionConflict(Exception):
    """Same idempotency key, different body."""


class IngestionInProgress(Exception):
    """Another worker holds the lease for this idempotency key."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS library_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestions (
    ingestion_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    request_id       TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    status           TEXT NOT NULL,
    visibility       TEXT NOT NULL,
    mode             TEXT NOT NULL,
    received_at      TEXT NOT NULL,
    committed_at     TEXT,
    lease_owner      TEXT,
    lease_expires_at TEXT,
    sync_ms          INTEGER,
    enrich_ms        INTEGER,
    UNIQUE (user_id, request_id)
);

-- One row per (user, session). ``status`` is the extraction state machine:
-- 'pending' once any chunk arrives, back to 'pending' when a session is
-- reopened by a later Add, 'extracted' after a finalisation attempt (even
-- one that produced zero cards -- without the attempt marker a noise-only
-- session would be re-extracted on every Search).
CREATE TABLE IF NOT EXISTS sessions (
    user_id            TEXT NOT NULL,
    session_id         TEXT NOT NULL,
    first_seen_at      TEXT NOT NULL,
    last_received_seq  INTEGER NOT NULL,
    chunk_count        INTEGER NOT NULL,
    status             TEXT NOT NULL,
    extraction_version TEXT,
    card_count         INTEGER NOT NULL DEFAULT 0,
    extracted_at       TEXT,
    PRIMARY KEY (user_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(user_id, status);

-- The chunk is the minimal persisted fact. ``message_index``/``role`` are
-- denormalised from the (not stored) messages so session timelines can be
-- reconstructed without a messages table: order is (Add order via
-- received_seq, message index, chunk position).
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    message_id      TEXT NOT NULL,
    message_index   INTEGER NOT NULL,
    role            TEXT NOT NULL,
    ingestion_id    INTEGER NOT NULL REFERENCES ingestions(ingestion_id) ON DELETE CASCADE,
    ord             INTEGER NOT NULL,
    char_start      INTEGER NOT NULL,
    char_end        INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    text            TEXT NOT NULL,
    parent_chunk_id TEXT,
    dl              INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    source_ts       INTEGER,
    ts_origin       TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    received_seq    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_message ON chunks(message_id, ord);
CREATE INDEX IF NOT EXISTS idx_chunks_ingestion ON chunks(ingestion_id);
CREATE INDEX IF NOT EXISTS idx_chunks_session
    ON chunks(user_id, ingestion_id, message_index, ord);

CREATE TABLE IF NOT EXISTS experiences (
    experience_id     TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    ingestion_id      INTEGER NOT NULL REFERENCES ingestions(ingestion_id) ON DELETE CASCADE,
    kind              TEXT NOT NULL,
    payload           TEXT NOT NULL,
    full_content      TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_user ON experiences(user_id);
CREATE INDEX IF NOT EXISTS idx_exp_session ON experiences(user_id, session_id);

CREATE TABLE IF NOT EXISTS experience_terms (
    user_id       TEXT NOT NULL,
    term          TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    tf            INTEGER NOT NULL,
    PRIMARY KEY (user_id, term, experience_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS experience_entities (
    user_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    value         TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (user_id, kind, value, experience_id)
) WITHOUT ROWID;

-- Structural anchors of a case packet: exact file paths, symbols and
-- conditions (version/module/error).  Query-time matching against
-- these is what links a current issue to a historical repair; the
-- card itself stores the same values as facts, never a verdict.
CREATE TABLE IF NOT EXISTS case_anchors (
    user_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    value         TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (user_id, kind, value, experience_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_case_anchors_lookup
    ON case_anchors(user_id, kind, value);

-- Session-scoped chunk windows join ingestions on (user_id, session_id);
-- without this the evidence/session scans are full table scans.
CREATE INDEX IF NOT EXISTS idx_ingestions_scope ON ingestions(user_id, session_id);

-- Card vectors: the only embedding store.  Chunks are never embedded; a
-- card carries the embedding of its own index face.  Written together with
-- the card row, keyed by the embedding model so a cross-model cosine (which
-- is not a similarity) can never be computed.
CREATE TABLE IF NOT EXISTS experience_vectors (
    user_id       TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    model         TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    vec           BLOB NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (user_id, experience_id, model)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_experience_vectors_user
    ON experience_vectors(user_id, model);
"""


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


class Store:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._stats_cache: dict[str, tuple[int, float]] = {}
        self._init_schema()

    # -- connection management -------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.settings.db_path,
            timeout=30.0,
            isolation_level=None,  # explicit transaction control
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is None:
            existing = self._connect()
            self._local.conn = existing
        return existing

    @contextmanager
    def write_tx(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction.

        SQLite admits one writer; serialising explicitly turns contention into
        a short wait instead of a SQLITE_BUSY error under concurrent Add.
        """
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # -- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        with self._write_lock:
            conn = self.conn
            self._refuse_older_library(conn)
            conn.executescript(_SCHEMA_SQL)
            conn.execute(
                "INSERT INTO library_info(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _refuse_older_library(conn: sqlite3.Connection) -> None:
        """Refuse a library whose shape this code cannot read or write.

        ``_init_schema`` only ever runs ``CREATE TABLE IF NOT EXISTS``, so it
        can add tables but never reshape an existing one.  Every version below
        v8 differs in column set or primary key (v8 drops the always-1 card
        ``version``), and a file that kept the old shape would fail deep inside
        an INSERT instead of here.  v9 drops the meta layer; v8 files are
        refused for the same reason.  Newer files are refused too: their
        columns may not exist in this code's statements.  The stamp is checked
        before it is rewritten, so a refused file is never relabelled as
        current.
        """
        version: str | None = None
        try:
            row = conn.execute(
                "SELECT value FROM library_info WHERE key = 'schema_version'"
            ).fetchone()
            version = str(row["value"]).strip() if row is not None else None
        except sqlite3.OperationalError:
            pass
        if version is None:
            # Pre-v9 files carried the stamp in a table named ``meta``.
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                version = str(row["value"]).strip() if row is not None else None
            except sqlite3.OperationalError:
                return  # brand-new file, no stamp table yet
        if version is None or version == str(SCHEMA_VERSION):
            return
        raise RuntimeError(
            f"memory library at schema v{version} cannot be opened by "
            f"schema v{SCHEMA_VERSION} code: the card tables changed "
            "shape and are created in place, not migrated; archive or delete "
            "the file and rebuild the library by replaying the Add stream"
        )

    # -- counters ---------------------------------------------------------

    def _next_seq(self, conn: sqlite3.Connection, name: str = "received_seq") -> int:
        conn.execute(
            "INSERT INTO counters(name, value) VALUES(?, 1) "
            "ON CONFLICT(name) DO UPDATE SET value = value + 1",
            (name,),
        )
        row = conn.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
        return int(row["value"])

    # -- ingestion lifecycle ----------------------------------------------

    def get_ingestion(self, user_id: str, request_id: str) -> IngestionState | None:
        row = self.conn.execute(
            "SELECT ingestion_id, user_id, request_id, session_id, content_hash, "
            "       status, visibility, mode "
            "FROM ingestions WHERE user_id = ? AND request_id = ?",
            (user_id, request_id),
        ).fetchone()
        if row is None:
            return None
        return IngestionState(
            ingestion_id=int(row["ingestion_id"]),
            user_id=row["user_id"],
            request_id=row["request_id"],
            session_id=row["session_id"],
            content_hash=row["content_hash"],
            status=row["status"],
            visibility=row["visibility"],
            mode=row["mode"],
        )

    def begin_ingestion(
        self,
        *,
        user_id: str,
        request_id: str,
        session_id: str,
        digest: str,
        mode: str,
        lease_seconds: float,
    ) -> IngestionState:
        """Claim the idempotency key, or report why we cannot.

        Reserves the key *before* any model work so concurrent duplicates
        cannot both proceed.
        """
        owner = uuid.uuid4().hex
        now = datetime.now(UTC)
        expires = (now + timedelta(seconds=lease_seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        with self.write_tx() as conn:
            row = conn.execute(
                "SELECT ingestion_id, content_hash, status, lease_expires_at, session_id, "
                "       visibility, mode "
                "FROM ingestions WHERE user_id = ? AND request_id = ?",
                (user_id, request_id),
            ).fetchone()

            if row is not None:
                if row["content_hash"] != digest:
                    raise IngestionConflict(
                        "request_id already used with a different request body"
                    )
                if row["status"] == "committed":
                    return IngestionState(
                        ingestion_id=int(row["ingestion_id"]),
                        user_id=user_id,
                        request_id=request_id,
                        session_id=row["session_id"],
                        content_hash=row["content_hash"],
                        status=row["status"],
                        visibility=row["visibility"],
                        mode=row["mode"],
                    )
                lease = row["lease_expires_at"]
                if row["status"] == "processing" and lease and lease > now.isoformat().replace("+00:00", "Z"):
                    raise IngestionInProgress(
                        "request_id is being processed by another worker"
                    )
                # Failed, or the lease expired: take it over.
                conn.execute(
                    "UPDATE ingestions SET status='processing', mode=?, lease_owner=?, "
                    "       lease_expires_at=?, session_id=?, received_at=? "
                    "WHERE ingestion_id = ?",
                    (mode, owner, expires, session_id, now_iso(), row["ingestion_id"]),
                )
                return IngestionState(
                    ingestion_id=int(row["ingestion_id"]),
                    user_id=user_id,
                    request_id=request_id,
                    session_id=session_id,
                    content_hash=digest,
                    status="processing",
                    visibility="lex_only",
                    mode=mode,
                )

            cursor = conn.execute(
                "INSERT INTO ingestions(user_id, request_id, session_id, content_hash, "
                "       status, visibility, mode, received_at, lease_owner, lease_expires_at) "
                "VALUES(?,?,?,?,'processing','lex_only',?,?,?,?)",
                (user_id, request_id, session_id, digest, mode, now_iso(), owner, expires),
            )
            return IngestionState(
                ingestion_id=int(cursor.lastrowid),
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                content_hash=digest,
                status="processing",
                visibility="lex_only",
                mode=mode,
            )

    def mark_failed(self, ingestion_id: int) -> None:
        with self.write_tx() as conn:
            conn.execute(
                "UPDATE ingestions SET status='failed', lease_owner=NULL, lease_expires_at=NULL "
                "WHERE ingestion_id = ? AND status != 'committed'",
                (ingestion_id,),
            )

    def commit_chunks(
        self,
        *,
        state: IngestionState,
        messages: Sequence[MessageInput],
        chunks: Sequence[ChunkRecord],
        sync_ms: int,
    ) -> None:
        """Sync commit: chunks (the persisted facts) plus session bookkeeping.

        Raw messages are intentionally not stored -- the chunk is the minimal
        fact unit, and chunks alone reconstruct every timeline the card layer
        needs. Everything lands in one transaction, so a committed ingestion
        always implies a complete, durable chunk snapshot.
        """
        with self.write_tx() as conn:
            seq = self._next_seq(conn)
            received = now_iso()

            for chunk in chunks:
                source_ts = messages[chunk.message_index].timestamp
                conn.execute(
                    "INSERT OR REPLACE INTO chunks(chunk_id, user_id, message_id, "
                    "       message_index, role, ingestion_id, ord, char_start, char_end, "
                    "       kind, text, parent_chunk_id, dl, content_hash, source_ts, "
                    "       ts_origin, received_at, received_seq) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.chunk_id,
                        state.user_id,
                        chunk.message_id,
                        chunk.message_index,
                        messages[chunk.message_index].role,
                        state.ingestion_id,
                        chunk.ord,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.kind,
                        chunk.text,
                        chunk.parent_chunk_id,
                        len(chunk.tokens),
                        hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                        source_ts,
                        "source" if source_ts is not None else "received",
                        received,
                        seq,
                    ),
                )

            # A reopened session goes back to pending so the next finalisation
            # re-extracts (and replaces) its cards with the fuller timeline.
            conn.execute(
                "INSERT INTO sessions(user_id, session_id, first_seen_at, "
                "       last_received_seq, chunk_count, status, card_count) "
                "VALUES(?,?,?,?,?,'pending',0) "
                "ON CONFLICT(user_id, session_id) DO UPDATE SET "
                "       last_received_seq=excluded.last_received_seq, "
                "       chunk_count=sessions.chunk_count + excluded.chunk_count, "
                "       status='pending'",
                (state.user_id, state.session_id, received, seq, len(chunks)),
            )

            conn.execute(
                "UPDATE ingestions SET status='committed', visibility='lex_only', "
                "       committed_at=?, lease_owner=NULL, lease_expires_at=NULL, sync_ms=? "
                "WHERE ingestion_id = ?",
                (received, sync_ms, state.ingestion_id),
            )
        self._stats_cache.clear()

    # -- session lifecycle ---------------------------------------------------

    def pending_sessions(self, user_id: str, exclude_session_id: str | None = None) -> list[str]:
        """Sessions with committed chunks but no completed extraction attempt."""
        if exclude_session_id is None:
            rows = self.conn.execute(
                "SELECT session_id FROM sessions WHERE user_id = ? AND status = 'pending' "
                "ORDER BY last_received_seq",
                (user_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT session_id FROM sessions WHERE user_id = ? AND status = 'pending' "
                "AND session_id != ? ORDER BY last_received_seq",
                (user_id, exclude_session_id),
            ).fetchall()
        return [row["session_id"] for row in rows]

    def last_session_id(self, user_id: str) -> str | None:
        """Session of the most recent ingestion, used for change detection."""
        row = self.conn.execute(
            "SELECT session_id FROM ingestions WHERE user_id = ? "
            "ORDER BY ingestion_id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return row["session_id"] if row is not None else None

    def last_ingestion_of_session(self, user_id: str, session_id: str) -> int | None:
        """Latest ingestion id of a session (owner of search-triggered cards)."""
        row = self.conn.execute(
            "SELECT ingestion_id FROM ingestions WHERE user_id = ? AND session_id = ? "
            "ORDER BY ingestion_id DESC LIMIT 1",
            (user_id, session_id),
        ).fetchone()
        return int(row["ingestion_id"]) if row is not None else None

    def mark_session_extracted(
        self,
        *,
        user_id: str,
        session_id: str,
        extraction_version: str,
        card_count: int,
    ) -> None:
        """Record a completed extraction attempt (even with zero cards)."""
        with self.write_tx() as conn:
            conn.execute(
                "UPDATE sessions SET status='extracted', extraction_version=?, "
                "       card_count=?, extracted_at=? "
                "WHERE user_id = ? AND session_id = ?",
                (extraction_version, card_count, now_iso(), user_id, session_id),
            )

    def session_cards(self, user_id: str, session_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT experience_id FROM experiences WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchall()
        return [row["experience_id"] for row in rows]

    def delete_session_experiences(
        self,
        user_id: str,
        session_id: str,
    ) -> int:
        """Drop a session's cards and their index rows.

        Called by finalisation before inserting the replacement cards, so a
        re-opened session cannot accumulate stale duplicates.
        """
        with self.write_tx() as conn:
            ids = [
                row["experience_id"]
                for row in conn.execute(
                    "SELECT experience_id FROM experiences "
                    "WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id),
                ).fetchall()
            ]
            for start in range(0, len(ids), 400):
                batch = ids[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(
                    f"DELETE FROM experiences WHERE user_id = ? "
                    f"AND experience_id IN ({placeholders})",
                    (user_id, *batch),
                )
                for table in ("experience_terms", "experience_entities",
                              "case_anchors", "experience_vectors"):
                    conn.execute(
                        f"DELETE FROM {table} WHERE user_id = ? "
                        f"AND experience_id IN ({placeholders})",
                        (user_id, *batch),
                    )
        return len(ids)

    # -- card vector channel ------------------------------------------------

    def store_experience_vectors(
        self,
        user_id: str,
        rows: Sequence[tuple[str, str, int, bytes]],
    ) -> int:
        """Persist (experience_id, model, dim, blob) card vectors.

        Same contract as ``store_vectors``: written after the card commit, in
        its own transaction, and a failure only means the card is searched
        lexically.
        """
        if not rows:
            return 0
        created = now_iso()
        with self.write_tx() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO experience_vectors"
                "(user_id, experience_id, model, dim, vec, created_at) "
                "VALUES(?,?,?,?,?,?)",
                [
                    (user_id, eid, model, dim, blob, created)
                    for eid, model, dim, blob in rows
                ],
            )
        return len(rows)

    def experience_vector_search(
        self,
        user_id: str,
        query_vector: Sequence[float],
        model: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Cosine similarity over the vectors of cards that still exist.

        Brute force, like ``vector_search``: a scope holds hundreds of cards,
        so a scan beats maintaining an ANN index.  The join is what keeps a
        vector whose card was deleted with its session out of the results.
        """
        import math
        import struct

        dim = len(query_vector)
        if dim == 0:
            return []
        qnorm = math.sqrt(sum(v * v for v in query_vector))
        if qnorm == 0:
            return []

        rows = self.conn.execute(
            "SELECT v.experience_id AS eid, v.vec AS vec "
            "FROM experience_vectors v "
            "JOIN experiences e ON e.user_id = v.user_id "
            "  AND e.experience_id = v.experience_id "
            "WHERE v.user_id = ? AND v.model = ? AND v.dim = ?",
            (user_id, model, dim),
        ).fetchall()

        scored: list[tuple[str, float]] = []
        for row in rows:
            blob = row["vec"]
            if len(blob) != dim * 4:
                continue
            vector = struct.unpack(f"<{dim}f", blob)
            dot = 0.0
            norm = 0.0
            for a, b in zip(query_vector, vector):
                dot += a * b
                norm += b * b
            if norm == 0:
                continue
            scored.append((row["eid"], dot / (qnorm * math.sqrt(norm))))

        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:limit]

    def experience_vector_coverage(self, user_id: str, model: str) -> dict:
        """How much of the card layer is vector-indexed."""
        total = self.conn.execute(
            "SELECT COUNT(*) AS n FROM experiences WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]
        indexed = self.conn.execute(
            "SELECT COUNT(*) AS n FROM experience_vectors WHERE user_id = ? AND model = ?",
            (user_id, model),
        ).fetchone()["n"]
        return {
            "cards": int(total),
            "vectorized": int(indexed),
            "coverage": round(indexed / total, 4) if total else None,
        }

    def recover_stale(self) -> int:
        """Reap ingestions whose lease expired mid-processing.

        Their partial work lives inside an uncommitted transaction, so there is
        nothing to roll back -- flipping the state is enough to make them
        retryable.
        """
        now = now_iso()
        with self.write_tx() as conn:
            cursor = conn.execute(
                "UPDATE ingestions SET status='failed', lease_owner=NULL, lease_expires_at=NULL "
                "WHERE status='processing' AND (lease_expires_at IS NULL OR lease_expires_at < ?)",
                (now,),
            )
            return cursor.rowcount or 0

    # -- experience layer ---------------------------------------------------
    #
    # A card is written exactly once and is never merged or rewritten, so an id
    # Search already returned keeps its meaning.  There is no version column:
    # it was pinned to 1 in every row it ever held, and a superseded card is
    # deleted with its session rather than re-versioned in place.

    def insert_experience(
        self,
        *,
        user_id: str,
        session_id: str,
        experience_id: str,
        ingestion_id: int,
        kind: str,
        payload_json: str,
        full_content: str,
        extraction_version: str,
        created_at: str,
        terms: Sequence[str],
        entities: Sequence[tuple[str, str]],
        anchors: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Persist one card and its complete search index.

        Cards are written by session finalisation, after the sync commit, so
        a card only ever *adds* retrievable memory.  The card row and its
        term/entity/anchor rows commit in the same transaction: a visible
        card always has a complete index.
        """
        with self.write_tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiences(experience_id, user_id, "
                "       session_id, ingestion_id, kind, payload, full_content, "
                "       extraction_version, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    experience_id,
                    user_id,
                    session_id,
                    ingestion_id,
                    kind,
                    payload_json,
                    full_content,
                    extraction_version,
                    created_at,
                ),
            )
            for table in ("experience_terms", "experience_entities", "case_anchors"):
                conn.execute(
                    f"DELETE FROM {table} WHERE user_id=? AND experience_id=?",
                    (user_id, experience_id),
                )
            for term, tf in term_frequencies(terms).items():
                conn.execute(
                    "INSERT INTO experience_terms(user_id, term, experience_id, tf) "
                    "VALUES(?,?,?,?)",
                    (user_id, term, experience_id, tf),
                )
            for entity_kind, entity_value in entities:
                conn.execute(
                    "INSERT INTO experience_entities(user_id, kind, value, experience_id) "
                    "VALUES(?,?,?,?)",
                    (user_id, entity_kind, entity_value, experience_id),
                )
            for anchor_kind, anchor_value in anchors:
                if not anchor_value:
                    continue
                conn.execute(
                    "INSERT INTO case_anchors(user_id, kind, value, experience_id) "
                    "VALUES(?,?,?,?)",
                    (user_id, anchor_kind, anchor_value.lower(), experience_id),
                )

    def experience_field_search(
        self,
        user_id: str,
        query: str | Sequence[str],
        limit: int,
        anchors: dict[str, Sequence[str]] | None = None,
    ) -> list[tuple[str, float]]:
        """Field-aware BM25 recall over the card index face.

        The card is indexed as ``field\\x1fterm`` (problem/locate/implement/
        test) instead of one concatenated blob.  Query errors are boosted on
        problem, paths/symbols on locate, test names on test; a plain legacy
        term is kept as a low-weight fallback for cards written before the
        field index existed.  Length normalisation uses the card's
        indexed term count so long cards do not win by verbosity alone.
        """
        if isinstance(query, str):
            raw_terms = list(term_frequencies(tokenize(query)))
        else:
            raw_terms = list(term_frequencies(t for token in query for t in tokenize(str(token))))
        raw_terms = list(dict.fromkeys(raw_terms))
        if not raw_terms:
            return []
        n_exp = self.conn.execute(
            "SELECT COUNT(*) AS n FROM experiences WHERE user_id = ?",
            (user_id,),
        ).fetchone()["n"]
        if n_exp == 0:
            return []

        field_weights = {
            "problem": self.settings.card_field_weight_problem,
            "locate": self.settings.card_field_weight_locate,
            "implement": self.settings.card_field_weight_implement,
            "test": self.settings.card_field_weight_test,
        }
        weights: dict[str, float] = {}

        def add_weight(field: str, term: str, weight: float) -> None:
            key = f"{field}\x1f{term}"
            weights[key] = max(weights.get(key, 0.0), weight)

        for term in raw_terms:
            for field, weight in field_weights.items():
                add_weight(field, term, weight)
            # Legacy plain terms (pre-field-index cards) still match weakly.
            weights[term] = max(weights.get(term, 0.0), 0.25)

        anchors = anchors or {}
        boost = self.settings.card_anchor_field_boost
        for kind, field in (
            ("error", "problem"),
            ("path", "locate"),
            ("symbol", "locate"),
            ("test_name", "test"),
        ):
            for value in anchors.get(kind, ()):
                for term in tokenize(str(value)):
                    add_weight(field, term, field_weights[field] * boost)

        placeholders = ",".join("?" for _ in weights)
        rows = self.conn.execute(
            f"SELECT et.experience_id AS eid, et.term AS term, et.tf AS tf "
            f"FROM experience_terms et "
            f"JOIN experiences e ON e.user_id = et.user_id "
            f"  AND e.experience_id = et.experience_id "
            f"WHERE et.user_id = ? AND et.term IN ({placeholders})",
            (user_id, *weights.keys()),
        ).fetchall()
        if not rows:
            return []

        import math

        # BM25 document length must be the length of the indexed document, not
        # the rendered packet. The index face includes field-prefixed terms
        # (and full patch/condition text), so SUM(tf) is the matching unit.
        dl_rows = self.conn.execute(
            "SELECT experience_id AS eid, SUM(tf) AS dl FROM experience_terms "
            "WHERE user_id = ? GROUP BY experience_id",
            (user_id,),
        ).fetchall()
        doc_len = {row["eid"]: max(1, row["dl"] or 1) for row in dl_rows}
        avgdl = (sum(doc_len.values()) / len(doc_len)) if doc_len else 1.0
        k1 = self.settings.card_bm25_k1
        b = self.settings.card_bm25_b

        df: dict[str, int] = {}
        for row in rows:
            df[row["term"]] = df.get(row["term"], 0) + 1
        scores: dict[str, float] = {}
        for row in rows:
            d = df.get(row["term"], 1)
            idf = math.log(1.0 + (n_exp - d + 0.5) / (d + 0.5))
            tf = float(row["tf"] or 1.0)
            dl = doc_len.get(row["eid"], avgdl)
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            bm25 = tf * (k1 + 1.0) / denom if denom else 0.0
            scores[row["eid"]] = scores.get(row["eid"], 0.0) + weights[row["term"]] * idf * bm25
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]

    def experience_entity_search(
        self,
        user_id: str,
        entity_values: Sequence[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Current cards sharing an exact entity with the query."""
        wanted = [value.lower() for value in entity_values if value]
        if not wanted:
            return []
        placeholders = ",".join("?" for _ in wanted)
        rows = self.conn.execute(
            f"SELECT ee.experience_id AS eid, COUNT(DISTINCT ee.value) AS hits "
            f"FROM experience_entities ee "
            f"JOIN experiences e ON e.user_id = ee.user_id "
            f"  AND e.experience_id = ee.experience_id "
            f"WHERE ee.user_id = ? AND ee.value IN ({placeholders}) "
            f"GROUP BY ee.experience_id ORDER BY hits DESC, ee.experience_id ASC LIMIT ?",
            (user_id, *wanted, limit),
        ).fetchall()
        return [(row["eid"], float(row["hits"])) for row in rows]

    def case_anchor_search(
        self,
        user_id: str,
        anchors: dict[str, Sequence[str]],
        limit: int,
    ) -> list[tuple[str, float]]:
        """IDF-weighted exact-anchor recall over current case packets.

        A file path or an error literal that appears in both a historical case
        and the current issue is the strongest transfer signal available to a
        memory service that cannot see the repository.  Common anchors (a
        module touched by every task) are down-weighted by IDF; rare ones (an
        unusual error string) dominate.
        """
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for kind, values in anchors.items():
            for value in values:
                normalized = str(value).strip().lower()
                key = (kind, normalized)
                if normalized and key not in seen:
                    seen.add(key)
                    pairs.append(key)
        if not pairs:
            return []

        n_cards = int(self.conn.execute(
            "SELECT COUNT(*) AS n FROM experiences WHERE user_id = ?",
            (user_id,),
        ).fetchone()["n"])
        if n_cards == 0:
            return []

        clauses = " OR ".join("(ca.kind=? AND ca.value=?)" for _ in pairs)
        params: list[object] = [user_id]
        for kind, value in pairs:
            params.extend((kind, value))
        rows = self.conn.execute(
            f"SELECT ca.kind AS kind, ca.value AS value, ca.experience_id AS eid "
            f"FROM case_anchors ca "
            f"JOIN experiences e ON e.user_id = ca.user_id "
            f"  AND e.experience_id = ca.experience_id "
            f"WHERE ca.user_id = ? AND ({clauses})",
            params,
        ).fetchall()
        if not rows:
            return []

        import math

        df: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (row["kind"], row["value"])
            df[key] = df.get(key, 0) + 1
        weights = {
            "file": 1.3,
            "symbol": 1.4,
            "error": 1.6,
            "issue": 1.1,
            "module": 1.0,
            "version": 1.0,
            "config": 0.8,
            "test_name": 1.2,
        }
        scores: dict[str, float] = {}
        for row in rows:
            key = (row["kind"], row["value"])
            d = df.get(key, 1)
            idf = math.log(1.0 + (n_cards - d + 0.5) / (d + 0.5))
            factor = weights.get(row["kind"], 1.0)
            scores[row["eid"]] = scores.get(row["eid"], 0.0) + idf * factor
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]

    def list_current_experience_ids(self, user_id: str, limit: int = 200) -> list[str]:
        """All current card ids in one scope, newest first.

        With card-only return mode, every card is a candidate; the ranking
        channels then decide which ones to show.  This is what makes the card
        layer's coverage independent of whether a card happens to share a term
        with the query.
        """
        rows = self.conn.execute(
            "SELECT experience_id FROM experiences WHERE user_id = ? "
            "ORDER BY created_at DESC, experience_id LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [row["experience_id"] for row in rows]

    def fetch_current_experiences(self, user_id: str, experience_ids: Sequence[str]) -> dict[str, dict]:
        """Payload for each card id, as {experience_id: {...}}."""
        if not experience_ids:
            return {}
        out: dict[str, dict] = {}
        for start in range(0, len(experience_ids), 400):
            batch = experience_ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT experience_id AS eid, payload, kind, created_at, full_content "
                f"FROM experiences WHERE user_id = ? AND experience_id IN ({placeholders})",
                (user_id, *batch),
            ).fetchall()
            for row in rows:
                out[row["eid"]] = {
                    "payload": row["payload"],
                    "kind": row["kind"],
                    "created_at": row["created_at"],
                    "full_content": row["full_content"],
                }
        return out

    def experience_coverage(self, user_id: str) -> dict[str, int]:
        """Coverage counters for diagnostics."""
        total = self.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]
        cards = self.conn.execute(
            "SELECT COUNT(*) AS n FROM experiences WHERE user_id = ?",
            (user_id,),
        ).fetchone()["n"]
        return {"chunks": int(total), "experience_cards": int(cards)}

    def fetch_experience_vectors(
        self,
        user_id: str,
        model: str,
    ) -> dict[str, list[float]]:
        """Card vectors of live cards, for one embedding model.

        Cluster scoring mixes vector cosine with lexical features, so rows
        written by a different model are excluded outright -- a cross-model
        cosine is not a similarity.  The join drops vectors whose card was
        deleted with its session.
        """
        import struct

        rows = self.conn.execute(
            "SELECT v.experience_id AS eid, v.dim AS dim, v.vec AS vec "
            "FROM experience_vectors v "
            "JOIN experiences e ON e.user_id = v.user_id "
            "  AND e.experience_id = v.experience_id "
            "WHERE v.user_id = ? AND v.model = ?",
            (user_id, model),
        ).fetchall()
        out: dict[str, list[float]] = {}
        for row in rows:
            dim = int(row["dim"] or 0)
            blob = row["vec"]
            if dim <= 0 or not blob or len(blob) != dim * 4:
                continue
            out[row["eid"]] = list(struct.unpack(f"<{dim}f", blob))
        return out

    # -- retrieval ---------------------------------------------------------

    def session_extraction_versions(self, user_id: str, session_id: str) -> set[str]:
        """Versions with persisted cards, for non-destructive extractor changes."""
        return {row[0] for row in self.conn.execute(
            "SELECT DISTINCT extraction_version FROM experiences WHERE user_id=? AND session_id=?",
            (user_id, session_id),
        )}

    def fetch_session_chunks(self, user_id: str, session_id: str, limit: int = -1) -> list[StoredChunk]:
        """All chunks of a session, in source order.

        ``received_seq`` only orders Add requests; every chunk committed by one
        Add shares it, so the timeline must be (Add order, message index,
        chunk position).  This is the window session finalisation extracts
        cards from.
        """
        rows = self.conn.execute(
            "SELECT chunk_id FROM chunks c "
            "JOIN ingestions i ON i.ingestion_id = c.ingestion_id "
            "WHERE c.user_id = ? AND i.session_id = ? "
            "ORDER BY c.received_seq, c.message_index, c.ord, c.char_start LIMIT ?",
            (user_id, session_id, limit),
        ).fetchall()
        ordered = [row["chunk_id"] for row in rows]
        if not ordered:
            return []
        fetched = self.fetch_chunks(user_id, ordered)
        return [fetched[cid] for cid in ordered if cid in fetched]

    def fetch_evidence_chunks(
        self,
        user_id: str,
        session_id: str,
        limit: int = -1,
    ) -> list[StoredChunk]:
        """Evidence-bearing chunks of one session, in source order.

        Card extraction only sees a bounded window, so the historical diff and
        the verification command often land outside it.  This returns the
        session's structural evidence (diff/command/test/log) regardless of the
        window, so a packet can stay self-contained without enlarging the model
        prompt.
        """
        rows = self.conn.execute(
            "SELECT c.chunk_id AS chunk_id "
            "FROM chunks c JOIN ingestions i ON i.ingestion_id = c.ingestion_id "
            "WHERE c.user_id = ? AND i.session_id = ? AND ("
            "  c.kind IN ('diff','command','test_result','stack','log') "
            "  OR c.text LIKE '%diff --git%' OR c.text LIKE '%+++ b/%' "
            "  OR c.text LIKE '%\"command\"%' OR c.text LIKE '%[工具调用]%'"
            ") ORDER BY c.received_seq, c.message_index, c.ord, c.char_start LIMIT ?",
            (user_id, session_id, limit),
        ).fetchall()
        ordered = [row["chunk_id"] for row in rows]
        if not ordered:
            return []
        fetched = self.fetch_chunks(user_id, ordered)
        return [fetched[cid] for cid in ordered if cid in fetched]

    def fetch_chunks(self, user_id: str, chunk_ids: Sequence[str]) -> dict[str, StoredChunk]:
        if not chunk_ids:
            return {}
        out: dict[str, StoredChunk] = {}
        # Chunked to stay well under SQLite's variable limit.
        for start in range(0, len(chunk_ids), 400):
            batch = chunk_ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT c.chunk_id, c.user_id, c.message_id, c.ord, c.char_start, c.char_end, "
                f"       c.kind, c.text, c.source_ts, c.ts_origin, i.session_id, c.role "
                f"FROM chunks c "
                f"JOIN ingestions i ON i.ingestion_id = c.ingestion_id "
                f"WHERE c.user_id = ? AND c.chunk_id IN ({placeholders})",
                (user_id, *batch),
            ).fetchall()
            for row in rows:
                out[row["chunk_id"]] = StoredChunk(
                    chunk_id=row["chunk_id"],
                    user_id=row["user_id"],
                    message_id=row["message_id"],
                    ord=int(row["ord"]),
                    char_start=int(row["char_start"]),
                    char_end=int(row["char_end"]),
                    kind=row["kind"],
                    text=row["text"],
                    source_ts=row["source_ts"],
                    ts_origin=row["ts_origin"],
                    session_id=row["session_id"],
                    role=row["role"],
                )
        return out

    def neighbour_chunks(
        self,
        user_id: str,
        message_id: str,
        ord_: int,
        window: int,
    ) -> list[StoredChunk]:
        """Adjacent chunks within one message (chunks never span messages)."""
        rows = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE user_id = ? AND message_id = ? "
            "AND ord BETWEEN ? AND ? AND ord != ? ORDER BY ord",
            (user_id, message_id, ord_ - window, ord_ + window, ord_),
        ).fetchall()
        return list(self.fetch_chunks(user_id, [row["chunk_id"] for row in rows]).values())

    # -- lifecycle ---------------------------------------------------------

    def health(self) -> dict[str, object]:
        try:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM ingestions WHERE status='committed'").fetchone()
            return {"db": "ok", "committed_ingestions": int(row["n"])}
        except sqlite3.Error as exc:  # pragma: no cover - operational path
            return {"db": "error", "detail": type(exc).__name__}

    def purge_user(self, user_id: str) -> dict[str, int]:
        """Delete everything for one isolation scope (retention obligation)."""
        removed: dict[str, int] = {}
        with self.write_tx() as conn:
            for table in (
                "experience_vectors",
                "case_anchors",
                "experience_entities",
                "experience_terms",
                "experiences",
                "chunks",
                "sessions",
                "ingestions",
            ):
                cursor = conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                removed[table] = cursor.rowcount or 0
        self._stats_cache.pop(user_id, None)
        return removed