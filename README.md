# Code Memory System (codemem)

Engineering-experience memory service for the **code track** of Agent Memory Challenge 2026.
The system implements only **Add** and **Search**; the platform performs Answer, Eval and publication.

It turns historical engineering sessions into a **card library**: sessions are chunked losslessly
into persisted facts, each closed session is distilled by `gpt-4o-mini` into at most three
experience cards, and Search returns **cards** — ranked historical evidence, never an answer.

> The service receives strings only: no repository access, no answer generation, no gold labels.
> Raw messages are never stored; the chunk is the minimal persisted fact, and only cards are
> ever returned.

## Repository layout

```
codemem/
├── common/     shared infrastructure: config, store (SQLite), chunking, embeddings,
│               llm client, request guard, token counting, casebook (deterministic packets)
├── add/        the Add algorithm: chunking-driven ingestion and card extraction
│   ├── attempt.py       card-v5 extractor (state/action/mechanism/result + evidence admission)
│   └── experience.py    versioned card model (card-v4/v5), admission, payload, index face
├── search/     the Search algorithm
│   ├── query.py         query feature extraction (entities, anchors, term groups)
│   ├── retrieval.py     4-channel recall + RRF fusion + deterministic floor
│   ├── rerank.py        external rerank client (fails soft)
│   └── applicability.py listwise applicability judge (soft, never drops)
├── service.py  MemoryService: orchestrates Add and Search behind the API
├── api.py      FastAPI app exposing /add, /search, /healthz
└── main.py     uvicorn entrypoint
```

## Storage form (schema v8)

| Layer | Stored? | Purpose |
| --- | --- | --- |
| Raw messages | **no** | chunked on arrival; bodies never land in the DB |
| Chunks | yes | the minimal facts; lossless segments, kept forever, evidence for cards |
| Cards | yes | the retrieval unit: `problem / locate / implement` + grounded structure (`files/symbols/conditions/pitfalls`) + deterministic `patch/verify`; ≤3 per session |
| Card vectors | yes | the **only** embedding store (`text-embedding-v4`, index face of the card) |
| Meta | yes | cross-session aggregated rules; cites ≥2 cards from ≥2 different sessions; no paths/patches |
| Chunk lexical index / chunk vectors | **no** | chunks are facts, not retrieval items |

## Add: card-v5, an evidence-bound attempt

The default extractor remains **card-v4**. `card-v5` adds a separate semantic format:
**state / action / mechanism / result**. One card is one meaningful attempt, including failed or
reverted attempts, not a retrospective combination of every edit and the session's final
successful test run.

- `state`: symptoms, inputs, configuration and code facts known before acting.
- `action`: the actual investigation or change, not an instruction for the new task.
- `mechanism`: the historical explanation/hypothesis; nullable when unsupported.
- `result`: exact historical observation quotes, including failures, incomplete validation and
  explicitly recorded untested scope.

Per-field `evidence` entries (`field`, `chunk_id`, `quote`) are stored as audit metadata. The
extractor cites evidence by **continuous line numbers** over the `L`-labelled prompt lines —
gpt-4o-mini cannot transcribe quotes reliably, so admission resolves the cited address to the
exact stored span and the model never transcribes — it only selects lines. Admission checks
exact, unambiguous quotations, user/session scope, field-specific code anchors and
state/action/result chronology. The model's `result` summary is **not trusted**: the stored
result is reconstructed from the result evidence in source order.

All v5 index channels are state-based: lexical terms and vectors use the state face; auxiliary
entities/anchors come only from state and its cited quotes. Actions, mechanisms, results and
whole-session evidence are excluded from v5 recall.

Per batch, the extractor draws several independent samples (`CODEMEM_V5_SAMPLE_CALLS`, default 2)
whose admitted cards are deduplicated into a union; when the union is empty while proposals
exist, up to two feedback repair calls re-run extraction with the labeled rejection reasons.
Enable with:

```powershell
$env:CODEMEM_EXPERIENCE_EXTRACTION_VERSION = "card-v5"
$env:CODEMEM_DB_PATH = "path\to\codemem-card-v5.db"   # use a separate library, never migrate in place
```

## Search: cards only

- **Card channels (4):** field-aware BM25 over the card index face
  (`problem/locate/implement/test` groups), exact card entities, IDF-weighted case anchors
  (files/symbols/errors), and the **card's own embedding**. Fusion is RRF over ranks (`k=60`).
- **Deterministic floor:** a card must reach `CODEMEM_MIN_SCORE_RATIO` (0.15) of the response's
  best score to enter `data` (with `min_output_items` kept regardless). Chunks are never
  returned as items.
- **Soft stages:** the card ranking's top **20** candidates go to listwise applicability
  (tri-state, gpt-4o-mini, soft) and then to the external reranker API — recall 20 → rerank 20 →
  floor + final cut (`qwen3-rerank` via `RERANK_*`, fusion `alpha=0.3`). Both may reorder but
  never drop, and both fail soft.

## Session lifecycle

The API offers no "session finished" signal, so a session is closed the moment an Add with a
different `session_id` arrives.

1. **Add (sync, zero LLM):** validate → idempotency reserve → chunk → commit chunks + session
   row (`pending`). `success: true` means persisted.
2. **Finalise (best-effort, gpt-4o-mini):** when an Add sees a *different* session_id, every
   still-pending older session is finalised — its whole timeline is extracted into ≤3 cards in
   bounded batches, the session's previous cards are **replaced** (idempotent for re-opened
   sessions), then cards are embedded.
3. **Search backstop:** before retrieval, every still-pending session of the scope is finalised,
   so the last session of an Add stream also gets cards.

## API contract

`POST /add`

```json
{
  "request_id": "eval:<run_id>:conv-0:chunk-0",
  "messages": [{"role": "user", "timestamp": 1704067200000, "content": "raw memory text"}],
  "user_id": "eval:<run_id>:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

Response echoes `request_id` / `user_id` / `session_id` with `success: true` once persisted and
immediately retrievable.

`POST /search`

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:<run_id>:conv-0",
  "top_k": 100
}
```

Response:

```json
{"data": [{"id": "mem_123", "content": "remembered fact text", "score": 0.87, "created_at": "2026-07-01T12:00:00Z"}]}
```

`options` is sent for multiple-choice questions only; `top_k` is fixed at 100 in external
evaluation. `data` is ranked by relevance and may be empty.

## Quick start

```bash
pip install -r requirements.txt
python -m uvicorn codemem.main:app --host 0.0.0.0 --port 8080
```

## Configuration

All configuration is environment-based; nothing is required for a local run. A `.env` file at
the repo root is loaded automatically (real environment variables always win). The full tunable
list lives in `codemem/common/config.py`; the operational keys cover the LLM/embedding endpoints,
`RERANK_*` (external reranker), `CODEMEM_*` (thresholds, batch sizes, extraction version), and
the API key guard.
