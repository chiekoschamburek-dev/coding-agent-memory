"""Storage schema.

Four layers, all keyed by ``user_id`` (the only hard isolation boundary):

* ``raw_message``  — L0, lossless and immutable. We receive the data once, so
  this is the only basis for provenance, audit, and later re-chunking.
* ``chunk``        — L1, deterministic structure-aware pieces of a message.
* ``chunk_entity`` — L2, deterministic identifiers per chunk (inverted index).
* ``memory``       — L3, the retrievable unit. A memory is either a raw chunk
  or (later) an LLM-extracted experience card. Search only ever sees this
  table, which keeps generation strictly inside Add.

``request_seen`` implements Add idempotency: retries keep the same
``request_id`` and must not duplicate memory.

``memory_fts`` is an FTS5 external-content index over ``memory``; triggers keep
it in sync so a transaction that inserts memory is immediately searchable,
which is what the contract requires before returning HTTP 200.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_seen (
    request_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    n_messages  INTEGER NOT NULL,
    n_memories  INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_request_seen_user ON request_seen(user_id);

CREATE TABLE IF NOT EXISTS raw_message (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    request_id   TEXT NOT NULL,
    msg_index    INTEGER NOT NULL,
    role         TEXT NOT NULL,
    ts           INTEGER,
    content      TEXT NOT NULL,
    content_sha  TEXT NOT NULL,
    chars        INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE (user_id, request_id, msg_index)
);
CREATE INDEX IF NOT EXISTS ix_raw_user_session ON raw_message(user_id, session_id);
CREATE INDEX IF NOT EXISTS ix_raw_sha ON raw_message(user_id, content_sha);

CREATE TABLE IF NOT EXISTS chunk (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    request_id   TEXT NOT NULL,
    msg_index    INTEGER NOT NULL,
    part_index   INTEGER NOT NULL,
    part_count   INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    lang         TEXT,
    text         TEXT NOT NULL,
    sha          TEXT NOT NULL,
    line_start   INTEGER NOT NULL,
    line_end     INTEGER NOT NULL,
    ts           INTEGER,
    ord          INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE (user_id, sha, request_id)
);
CREATE INDEX IF NOT EXISTS ix_chunk_user ON chunk(user_id);
CREATE INDEX IF NOT EXISTS ix_chunk_user_kind ON chunk(user_id, kind);

CREATE TABLE IF NOT EXISTS chunk_entity (
    chunk_id   INTEGER NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
    user_id    TEXT NOT NULL,
    etype      TEXT NOT NULL,
    value_norm TEXT NOT NULL,
    value_raw  TEXT NOT NULL,
    PRIMARY KEY (chunk_id, etype, value_norm)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_entity_lookup ON chunk_entity(user_id, etype, value_norm);
CREATE INDEX IF NOT EXISTS ix_entity_user ON chunk_entity(user_id);

-- L3: the only table Search reads. kind = 'chunk' | 'card' | 'episode'.
-- ``sparse`` carries identifier-expanded tokens (snake/camel subtokens) so
-- FTS5 can match code identifiers that the unicode61 tokenizer would keep as
-- one indivisible token.
CREATE TABLE IF NOT EXISTS memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    request_id  TEXT NOT NULL,
    chunk_id    INTEGER REFERENCES chunk(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    title       TEXT,
    text        TEXT NOT NULL,
    sparse      TEXT NOT NULL DEFAULT '',
    sha         TEXT NOT NULL,
    ts          INTEGER,
    ord         INTEGER NOT NULL,
    superseded_by INTEGER REFERENCES memory(id),
    created_at  TEXT NOT NULL,
    UNIQUE (user_id, sha)
);
CREATE INDEX IF NOT EXISTS ix_memory_user ON memory(user_id);
CREATE INDEX IF NOT EXISTS ix_memory_user_ts ON memory(user_id, ts);
CREATE INDEX IF NOT EXISTS ix_memory_chunk ON memory(chunk_id);
CREATE INDEX IF NOT EXISTS ix_memory_kind ON memory(user_id, kind);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    text,
    title,
    sparse,
    content='memory',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, text, title, sparse)
    VALUES (new.id, new.text, coalesce(new.title, ''), new.sparse);
END;
CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, text, title, sparse)
    VALUES ('delete', old.id, old.text, coalesce(old.title, ''), old.sparse);
END;
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, text, title, sparse)
    VALUES ('delete', old.id, old.text, coalesce(old.title, ''), old.sparse);
    INSERT INTO memory_fts(rowid, text, title, sparse)
    VALUES (new.id, new.text, coalesce(new.title, ''), new.sparse);
END;

-- Entity -> time-ordered memories, so recency can weight rather than filter.
CREATE TABLE IF NOT EXISTS entity_timeline (
    user_id    TEXT NOT NULL,
    etype      TEXT NOT NULL,
    value_norm TEXT NOT NULL,
    memory_id  INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    ts         INTEGER,
    weight     REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (user_id, etype, value_norm, memory_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_timeline_lookup ON entity_timeline(user_id, etype, value_norm);

-- Derived repository identity, used as a soft filter and as a diagnostic for
-- whether one user_id actually spans several repositories.
CREATE TABLE IF NOT EXISTS repo_profile (
    user_id    TEXT NOT NULL,
    etype      TEXT NOT NULL,
    value_norm TEXT NOT NULL,
    n          INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, etype, value_norm)
) WITHOUT ROWID;

-- Content-hash cache for LLM enrichment, so retries and repeated text never
-- pay twice.
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key  TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Soft forgetting: an old memory superseded by a newer one is down-weighted,
-- never deleted (we cannot know the repository's current state).
CREATE TABLE IF NOT EXISTS supersedes (
    old_memory_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    new_memory_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    reason        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (old_memory_id, new_memory_id)
) WITHOUT ROWID;

-- Dense vectors kept beside the row id; loaded into FAISS at startup.
CREATE TABLE IF NOT EXISTS memory_vector (
    memory_id INTEGER PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
    user_id   TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_vector_user ON memory_vector(user_id);
"""
