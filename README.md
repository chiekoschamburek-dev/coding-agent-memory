# Code Memory System (codemem)

Engineering-experience memory service for the **code track** of Agent Memory
Challenge 2026. The system implements **Add** and **Search**; the platform
performs Answer, Eval and publication.

This repository contains the algorithm extracted from `D:\桌面\project\memory`,
synchronised on 2026-10-31. It retains the `add / search / common` package layout.
Source modules are preserved with import-path adaptations and a repository-root
path adjustment in `common/config.py`. Experiment scripts, datasets, databases
and credentials are outside this repository's scope.

Incoming messages are chunked losslessly into persisted facts. Sessions
are distilled by `gpt-4o-mini` into experience cards; Search returns ranked
historical evidence. Raw message bodies are never stored as separate records.
Cards are preferred; optional pending-session fallback can return persisted
chunk excerpts until extraction succeeds.

## Repository layout

```text
codemem/
├── common/                 shared infrastructure
│   ├── config.py           environment settings and repository paths
│   ├── store.py            SQLite schema v9, sessions, cards and indexes
│   ├── chunking.py         lossless chunks and entity extraction
│   ├── embeddings.py       embedding client and model concurrency limiter
│   ├── embedding_cache.py  SQLite cache, byte-bounded LRU and legacy import
│   ├── casebook.py         deterministic card-v4 repair packets
│   └── llm.py / guard.py / tokens.py
├── add/
│   ├── experience.py       versioned card interface, extraction and admission
│   └── minimal.py          card-v5-minimal evidence, admission and rendering
├── search/
│   ├── query.py            query features, entities and anchors
│   ├── retrieval.py        card recall, RRF fusion and output admission
│   ├── applicability.py    soft listwise applicability judge
│   ├── transferability.py minimal-card hypothesis and batch grouping
│   └── rerank.py           external rerank client
├── service.py              Add / Search orchestration
├── api.py                  FastAPI endpoints
└── main.py                 uvicorn entrypoint
tests/                      adapted algorithm and API regression suite
```

## Storage and card formats

Schema v9 stores chunks, session extraction status/version, cards, card lexical
indexes and card vectors. Chunks are evidence rather than embedding/retrieval
units, except for the optional pending-session fallback.

| Version | Shape | Use |
| --- | --- | --- |
| `card-v4` | `problem / locate / implement`, grounded structure and deterministic `patch / verify` | default; at most three repair cards per session |
| `card-v5-minimal` | `problem / mechanism / verify` | opt-in; one idea-level card per session |

`card-v4` grounds paths, symbols and conditions in cited chunks. Patch and
verification packets are derived from the evidence rather than accepted from
model output.

`card-v5-minimal` describes the historical cause and observable acceptance
semantics. The evidence window prefers the closing summary, removes fix
sections, and keeps testing evidence separately. If no summary exists, the
summary fallback (enabled by default) reconstructs evidence around the final fix
diff. Admission rejects paths, line numbers, commands, diffs and fix instructions;
code identifiers must occur in the evidence. Cards that exceed the render
budget are omitted rather than truncated.

Use a separate database for each extraction version. Finalisation refuses to
replace existing cards across a version boundary.

## Session lifecycle

Set `CODEMEM_ADD_FINALIZE_SYNC=true` when each Add contains a complete session:

1. **Add:** validate, reserve the idempotency key, chunk and persist session input.
2. **Extract:** wait for this session's extraction within its deadline, including
   time spent waiting for a shared model slot.
3. **Publish:** atomically replace cards, indexes and session extraction state.
   Embed cards outside the write transaction, then acknowledge success.
4. **Retry:** incomplete extraction returns retryable HTTP 503. Replaying the same
   request retries extraction without appending the committed chunks again.
5. **Search:** read published cards only; do not extract pending sessions or
   inject pending raw excerpts. A valid zero-card extraction also completes Add.

Same-session mutations are serialised. Distinct sessions, including those of
one user, extract concurrently within four shared model slots by default.
Atomic publication and WAL recall snapshots prevent partially replaced cards;
Search can read the previous complete publication during replacement.

The Python default for synchronous finalisation is `false`, retaining the legacy
lifecycle: an Add with a different `session_id` finalises older pending sessions,
and Search finalises pending sessions as a backstop. Optional background
finalisation and pending-session excerpt fallback apply only in this legacy mode.

## Search

Four card channels provide field-aware BM25, exact entities, IDF-weighted
anchors and card embeddings (`text-embedding-v4`). Request `options` can supply
a de-weighted probe. Reciprocal-rank fusion uses `k=60`; a deterministic score
floor controls admission, followed by content budgets and `top_k` (at most 100).
The default response ceiling is 10 memories (`CODEMEM_SEARCH_MAX_ITEMS`).

The applicability judge and external reranker are soft, fail-open stages.
Minimal cards must earn a channel hit, receive no `min_output_items` padding,
and skip the legacy repair judge.

The optional minimal-card transferability policy replaces the external reranker.
It analyses the issue once and judges at most 20 cards in one batch. Groups are
transferable first; topic-related, unknown and unjudged in the middle; irrelevant
last. It keeps candidates and their admission scores. Transferable results carry
a separately labelled, unverified query-time hypothesis and issue quote, without
changing stored cards. The model selects numbered evidence spans supplied by the
program; IDs are checked against the current issue or owning card and the program
fills in the original text. Diagnostics record stage timing, finish reasons and
rejection counts without source text. Grouped API responses omit `score`; failures
preserve baseline ordering. The two calls share a 60-second budget by default.

## API contract

`POST /add`

```json
{
  "request_id": "eval:run:conv-0:chunk-0",
  "messages": [{"role": "user", "timestamp": 1704067200000, "content": "raw memory text"}],
  "user_id": "eval:run:conv-0",
  "session_id": "eval:run:sample:0"
}
```

The response echoes the identifiers with `success: true` after persistence;
synchronous finalisation also requires completed extraction before success.

`POST /search`

```json
{
  "query": "Which historical experience matches this issue?",
  "options": ["A. First choice", "B. Second choice"],
  "user_id": "eval:run:conv-0",
  "top_k": 100
}
```

```json
{"data": [{"id": "mem_123", "content": "historical evidence", "score": 0.87, "created_at": "2026-07-01T12:00:00Z"}]}
```

`options` is optional. Results may be empty; `score` and `created_at` are optional.
Health probes are available at `/health` and `/healthz`.

## Quick start

```bash
pip install -r requirements.txt
python -m uvicorn codemem.main:app --host 0.0.0.0 --port 8080
```

All settings are environment-based. A repository-root `.env` is loaded without
overriding existing environment variables. The complete setting list is in
`codemem/common/config.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEMEM_EXPERIENCE_EXTRACTION_VERSION` | `card-v4` | select `card-v4` or `card-v5-minimal` |
| `CODEMEM_DB_PATH` | repository-root `data/codemem.db` | scoped memory store |
| `CODEMEM_MINIMAL_SUMMARY_FALLBACK` | `true` | reconstruct minimal evidence without a summary |
| `CODEMEM_ADD_FINALIZE_SYNC` | `false` | wait for extraction before acknowledging Add; disable background/backstop/raw fallback |
| `CODEMEM_FINALIZE_BACKGROUND` | `false` | legacy background finalisation, inactive in synchronous mode |
| `CODEMEM_SEARCH_PENDING_FALLBACK` | `false` | legacy pending-session excerpts, inactive in synchronous mode |
| `CODEMEM_SEARCH_MAX_ITEMS` | `10` | response ceiling, also bounded by `top_k` and content budgets |
| `CODEMEM_TRANSFERABILITY_ENABLED` | `false` | enable minimal-card transferability grouping |
| `CODEMEM_EMBED_CACHE_MEMORY_MB` | `16` | in-memory embedding-cache byte budget |

Configure `CODEMEM_LLM_*`, `CODEMEM_EMBED_*`, `RERANK_*` and `CODEMEM_API_KEY`
for model endpoints and API authentication. For transferability grouping, allow
enough search time for its model-call budget, for example
`CODEMEM_SEARCH_HARD_BUDGET_S=90`.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers extraction, minimal-card admission, session concurrency,
transferability, embedding caching, retrieval, API contracts and scope isolation.
Tests for offline library/view/index-building scripts are excluded because those
scripts are outside the extracted algorithm repository.

Verification on Windows (2026-10-31): **305 passed**. This includes synchronous
Add acknowledgement/retry, same-session deduplication, concurrent session
extraction, atomic publication, recall snapshots and evidence-ID validation.
