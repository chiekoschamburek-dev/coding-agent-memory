# codemem

A code-memory service implementing the **Add / Search** contract of the Agent
Memory Challenge (Coding track). The system stores engineering trajectories from
a repository and later retrieves the compact evidence needed to answer a
question about that repository under noisy conditions.

We implement only Add and Search; the platform runs Answer and Eval.

## What makes this different from a generic RAG stack

The Coding track measures retrieval under *relevant and noisy* conditions, where
every distractor comes from the same repository and therefore shares
vocabulary, file paths, and coding style. Three consequences drive the design:

1. **Structure before size.** Trajectories mix prose, diffs, stack traces, logs,
   commands, tests and config. Chunking follows those boundaries and never cuts
   a code fence mid-line, so a diff or a traceback survives intact.
2. **Identifiers are the load-bearing signal.** A file path, exception class, or
   symbol is far more discriminative than sentence similarity — and unlike
   embeddings, it does not drift when the pool is full of same-repo distractors.
   Identifiers are extracted deterministically and weighted by IDF, so a shared
   `src/parser/tokenizer.py` is weaker evidence than a shared `IndexError`.
3. **Not filling the budget is the point.** The platform feeds the answer model a
   token-counted prefix of our ranked output. Padding 100 slots with same-repo
   noise displaces the real evidence. A noise gate therefore stops early and
   prefers returning `[]` over returning distractors.

## Design invariants

1. **All content generation happens during Add.** Search only scores, filters,
   orders, and formats memory that already exists. Experience cards are produced
   at Add time, when no question exists yet — so they cannot encode an answer.
   Search never generates text.
2. **`user_id` is the only hard isolation boundary**, enforced on every read and
   write path, including the entity and recency channels and the FTS index.
3. **Add never fails because enrichment failed.** Raw text is persisted and
   committed before any LLM work, so a timeout degrades quality, not success.

## Architecture

```
Add   validate → idempotency → chunk (deterministic) → extract identifiers
      → persist + commit → [enrich: cards/episodes, optional, cached] → 200

Search  query plan → multi-channel recall → RRF fuse → score + gate
        → assemble evidence pyramid → data[]
```

Storage is a single SQLite database (WAL + FTS5), so one container is a complete
deployment with no external services.

| Layer | Contents | LLM? | Purpose |
|---|---|---|---|
| L0 `raw_message` | messages verbatim | no | we receive data once; basis for audit and re-chunking |
| L1 `chunk` | structure-aware pieces | no | retrieval granularity |
| L2 `chunk_entity` | paths, symbols, exceptions, tests, commands, packages, issues | no | exact-match retrieval signal |
| L3 `memory` | the retrievable unit (chunk, and later card/episode) | cards only | compact, judgeable evidence |
| governance | `entity_timeline`, `repo_profile`, `supersedes` | no | recency weighting, repo identity, soft forgetting (scaffolded) |

## Quick start

```bash
pip install -e ".[dev]"
python -m codemem            # serves on 0.0.0.0:8080
```

Or with Docker:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

### Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/add` | yes | store messages; 200 only once memory is searchable |
| `POST` | `/search` | yes | return ranked memory evidence |
| `GET` | `/health` | no | liveness; any 2xx |
| `DELETE` | `/admin/users/{user_id}` | yes | erase all trace of a user_id (retention) |

Authentication accepts `Authorization: Bearer <key>`, `Authorization: Token
<key>`, or `X-Api-Key`. Set `CODEMEM_API_KEY` to enable it; leaving it unset runs
unauthenticated, which the rules permit only for public smoke.

Business errors always use `{"detail": {"reason": "..."}}`.

## Configuration

Every tunable is an environment variable, which is what keeps the image
host-agnostic.

| Variable | Default | Notes |
|---|---|---|
| `CODEMEM_DATA_DIR` | `./data` | SQLite location; mount a volume in production |
| `CODEMEM_PORT` | `8080` | |
| `CODEMEM_API_KEY` | unset | unset = no auth (smoke only) |
| `CODEMEM_LLM_ENABLED` | `false` | enrichment is optional |
| `CODEMEM_LLM_BASE_URL` | unset | OpenAI-compatible endpoint |
| `CODEMEM_LLM_MODEL` | `gpt-4o-mini` | fixed by the rules for open-source entries |
| `CODEMEM_MIN_EVIDENCE_SCORE` | `0.15` | noise gate; calibrate with `eval/` |
| `CODEMEM_EVIDENCE_BUDGET_TOKENS` | `60000` | total payload budget, well under the 117,760 input window |

## Tests

```bash
pytest -q
```

Covers the contract (echoed identifiers, always-present `data`, the `top_k`
ceiling, error envelope), isolation (including the entity and recency channels),
idempotency, Add degradation, ranking under same-repo noise, and concurrency.

`tests/test_ranking_scale.py` deserves a note: it uses a few hundred
same-repository distractors, because several real defects were invisible at small
corpus size and only appeared at realistic scale.

### Diagnostics

`scripts/` holds the measurement tools used to find those defects, kept because
they are how the tuning decisions were made rather than guessed:

| Script | Answers |
|---|---|
| `characterize_gate.py` | which queries the noise gate admits, with one memory |
| `characterize_gate_corpus.py` | the same, but against hundreds of same-repo distractors, reporting the relevant memory's *rank* |
| `diagnose_lexical.py` | per-term document frequency and which candidates beat the relevant one |
| `diagnose_channel_attribution.py` | each channel ablated, to attribute a ranking failure |
| `probe_provider.py` | whether the LLM endpoint behaves like the required model |
| `loadtest.py` | latency and correctness under concurrent Add/Search |

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — data model, retrieval design, and the
  reasoning behind each invariant.
- [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) — mapping to the rules and the
  eight-item pre-Full checklist.
- [`deploy/README.md`](deploy/README.md) — public ingress options, since the
  platform requires a self-hosted publicly reachable API.

## Attribution

See [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) for third-party components,
their licences, and the disclosure of what is original work here.
