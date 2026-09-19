# Compliance

Mapping from the competition rules to concrete implementation, plus the
disclosure and attribution material the open-source division requires.

## Integrity rules (Section 8)

### 1. Return memory evidence only

Search must not generate a final answer and must not disguise an answer as a
memory record.

Enforcement is structural, not procedural, and is covered by
`tests/test_traceability.py` — which fails if returned content stops being a
verbatim span of what was Added:

- **`data[].content` is a verbatim slice of stored memory text.** Truncation to
  fit the token budget is selection and is allowed; rewriting is not. The contract
  states returned content is "preserved verbatim for audit", which is only
  meaningful if an auditor can locate our output in the input. A compact summary
  asserting something the trajectory never said would be fabricated evidence
  rather than a digest of it;
- **`created_at` carries the platform's source timestamp**, falling back to our
  persistence time only when the source had none. The two are never conflated: a
  processing clock is a fact about our pipeline that an auditor cannot find in
  the input;
- **no packaging is added to the text** — no ``[field] value`` headers, no
  separators, no invented timestamps inside ``content``. Identifiers are kept as
  retrieval keys in the index rather than duplicated into the payload;
- **the ``superseded`` marker travels on ``id``**, not in the content, because it
  is our judgement rather than memory text;
- **no free-form generation anywhere in the Search path**; the only LLM use
  planned there is *scoring* existing memories, which produces a number and never
  text returned to the platform;
- every returned item is a row in `memory`, written during Add;
- Search's assembly step composes a header from identifiers already extracted at
  Add time plus a verbatim excerpt — it has no code path that produces new
  prose, and no LLM call;
- enrichment that creates compact experience cards runs during Add, when no
  question exists yet, so a card cannot encode an answer to a question that has
  not been asked;
- `options` are used only to widen what the question is about. We deliberately do
  not perform per-option selection ("which memory supports option B"), which
  would amount to answering via retrieval ordering;
- `tests/test_governance.py` asserts returned content is traceable to stored
  text, that no option label is echoed, and that no answer framing appears.

The one place an LLM is permitted in the Search path (planned for P3) is
*scoring* existing memories — a relevance judgement over already-stored content,
which produces a number, never text returned to the platform.

### 2. Sample isolation

`user_id` is the only isolation boundary and it is enforced by API shape: every
storage method that reads memory takes `user_id` and filters on it. There is no
method that retrieves memory without one. This includes the channels most likely
to leak — identifier matching and recency, which are query-independent and would
otherwise be easy to get wrong — and the derived structures (`repo_profile`,
`entity_timeline`) and the FTS index. Verified in `tests/test_isolation.py`,
including a check that two users with byte-identical text remain separate.

### 3. Disclose sources and changes

See Attribution below.

### 4. No evaluation manipulation

- No hardcoding of benchmark content, and no dataset access — the task data is
  not public.
- No prompt injection: the only prompts are internal enrichment prompts, and all
  enrichment output is stored as memory, never used to influence scoring of
  other entries.
- No result manipulation: scores are strictly derived from stored content and
  configuration; ordering and scores are forced to agree.
- No live human answering: the service is fully automated.
- No benchmark leakage: our local evaluation uses public data (see `eval/`),
  kept out of the image and never redistributed.

## Full evaluation checklist (Section 6)

| # | Item | Status |
|---|---|---|
| 1 | Smoke passes; Add/Search usable | Achieved with P1; run before Full |
| 2 | API contract per official Add/Search format | See `tests/test_contract.py` |
| 3 | Add/Search model is `gpt-4o-mini` | Default `CODEMEM_LLM_MODEL=gpt-4o-mini`, fixed for the open-source entry. Enrichment uses it; Search uses it only for relevance scoring, never generation. |
| 4 | API reachable publicly for >= 30 days | Requires ingress (see `deploy/README.md`); image is host-agnostic |
| 5 | Complete run instructions | `README.md` plus `deploy/README.md` |
| 6 | Originality disclosed | This file |
| 7 | Substantive submission | Original data model, chunking, IDF identifier channel, gating and evidence assembly |
| 8 | No manipulation | Section 4 above |

### Note on the embedding model

The rules fix the **Add/Search model** at `gpt-4o-mini`. Our P1 baseline uses no
model at all: chunking, identifier extraction, BM25 and fusion are deterministic
and model-free. The dense channel (P2) is planned to use an open-source
*embedding* model, not a generative one, and `text-embedding-3-small` is
available as a drop-in switch via `CODEMEM_EMBED_BACKEND`. The position taken is
that a non-generative open-source encoder is not a "model" in the sense the rule
constrains, since it cannot generate an answer; this is stated explicitly here
so the choice is disclosed rather than assumed. If the organisers read the rule
more narrowly, the switch keeps the entry compliant.

## Attribution

### Original work in this repository

- The four-layer memory model (raw / chunk / entity / memory) and the governance
  structures (`entity_timeline`, `repo_profile`, `supersedes`).
- Structure-aware chunking that classifies fenced content by what it contains
  rather than by its fence label, and never splits a code fence mid-line.
- The deterministic identifier vocabulary and its guards (version strings,
  status-code-like tokens, prose path mentions).
- IDF-weighted identifier matching as a first-class recall channel, motivated by
  same-repository noise.
- The eligibility rule that recency alone cannot qualify a memory, and the
  damped identifier score that prevents distractors from winning on shared paths.
- The evidence pyramid and its token budget discipline.

### Third-party components

| Component | Role | Licence |
|---|---|---|
| FastAPI | HTTP framework | MIT |
| Pydantic | Request/response validation | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| SQLite / FTS5 | Storage and lexical index | Public domain |
| PyYAML | Configuration parsing | MIT |
| pytest, httpx | Tests | MIT / BSD-3-Clause |
| `tiktoken` (optional) | Token accounting | MIT |
| BGE-M3 (planned, P2) | Dense embeddings | MIT |
| bge-reranker-v2-m3 (planned, P2) | Cross-encoder reranking | MIT |
| FAISS (planned, P2) | Vector index | MIT |

BGE-M3 and bge-reranker are published by the BAAI FlagEmbedding project. FAISS
is published by Meta AI Research. If any of these are added, their model cards
and licences are to be cited here before submission.

### Methods referenced (not copied)

- Reciprocal Rank Fusion (Cormack, Clarke, Buettcher, 2009) — standard rank
  fusion, reimplemented.
- BM25 (Robertson & Zaragoza) as provided by SQLite FTS5.
- The general "memory system" framing common to Mem0, Zep, and similar systems;
  no code from those projects is used.

### Evaluation data

Local evaluation is intended to use public datasets (SWE-bench Lite / Verified,
MIT-licensed, and similar) purely to construct a proxy corpus of repository
trajectories. Such data is used for internal development only: it is excluded
from the Docker image, never redistributed, and never used for training,
fine-tuning, or dataset reconstruction.

## Data handling

- Only the current task's memory fragments, user/session identifiers, and the
  question are received; no gold answers or scoring criteria are requested.
- Request bodies are not logged. Logs carry identifiers, sizes, and timings.
- `DELETE /admin/users/{user_id}` erases raw messages, chunks, entities,
  memories, vectors, request records, profiles, timelines, and supersede links,
  and optimizes the FTS index.
- Data is not used for training, fine-tuning, product analysis, dataset
  reconstruction, or redistribution.
- Retention target: deletion within 30 days of task completion.
