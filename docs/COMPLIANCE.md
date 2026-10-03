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

The one place an LLM appears in the Search path is *scoring* existing memories
(`codemem.listwise`): a relevance judgement over already-stored content which
produces numbers, never text returned to the platform. It is implemented and off
by default. `tests/test_listwise.py` asserts that even a reply containing prose
contributes scores only, and that a malformed reply leaves the ranking unchanged.

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
| 4 | API reachable publicly for >= 30 days | **Hard blocker; not yet done.** Every track must self-host (see Deployment obligations below). The image is host-agnostic, so this is an infrastructure task, not a code task. |
| 5 | Complete run instructions | `README.md` plus `deploy/README.md` |
| 6 | Originality disclosed | This file |
| 7 | Substantive submission | Original data model, chunking, IDF identifier channel, gating and evidence assembly |
| 8 | No manipulation | Section 4 above |

## Deployment obligations (2026-09-26 rule update)

The organisers now require a self-hosted, publicly reachable Add/Search API for
**every** track; a repository or a Docker image is disclosure material and does
not substitute for a deployed endpoint, and the platform does not deploy entries
on a participant's behalf. The old "submit code, the platform builds it" path is
withdrawn, so items below are prerequisites for submission rather than
nice-to-haves.

- **Availability window.** The endpoint must stay reachable for at least 30 days
  after submission, and `full` runs are scheduled one per three months, so the
  window covers the queue as well as the evaluation. Tiers that sleep when idle
  or expire on a trial clock do not satisfy this.
- **Nothing may sit in front that can time out the request.** Add is synchronous
  and our internal guard is 1 500 s against a 30-minute contract ceiling
  (`Settings.add_deadline_seconds`), which means a worst-case Add can be a
  25-minute connection that sends nothing back. Any L7 proxy whose
  origin-response timeout is capped below that will return a gateway error and
  the platform will record a failure. Concretely: Alibaba Cloud ESA caps the
  full-chain back-to-origin timeout at 300 s (default 30 s, documented guidance
  <= 60 s), so it cannot front this service regardless of configuration; its edge
  functions are a JavaScript runtime, so it cannot host it either. Exposing the
  container port directly on a VM avoids the whole class of problem.
- **Authentication becomes mandatory**, not optional. The platform accesses the
  interface with a Memory System Key we issue, which maps to `CODEMEM_API_KEY`.
  Unauthenticated operation is permitted only for public smoke, and the public
  surface here includes `DELETE /admin/users/{user_id}`, which erases a user's
  memory permanently -- so a deployed instance without a key is a data-destruction
  endpoint open to the internet.
- **Capacity and timeout limits must be stated** in the submission: the contract
  limits we enforce (`max_messages_per_add=5000`, `max_content_chars=4 000 000`,
  `max_top_k=1000`, the 1 500 s Add guard), measured Add/Search latency, and the
  fact that writes are serialized while reads proceed concurrently under WAL.
- **Data obligations survive hosting.** Evaluation data and derived copies must be
  deleted within 30 days of task completion and may not be used for training,
  fine-tuning, product analysis, dataset reconstruction or redistribution; request
  bodies are not logged. Where the request path crosses a third party that logs
  or inspects bodies (a CDN, a WAF, or a third-party hosting platform), that
  hosting choice has to be assessed against these obligations, not assumed
  compliant.

### Note on the encoder models

The rules fix the **Add/Search model** at `gpt-4o-mini`. Our deterministic
baseline uses no model at all: chunking, identifier extraction, BM25 and fusion
are model-free. The dense and rerank channels ship with non-generative
open-source encoders (BAAI/bge-small-en-v1.5 and
cross-encoder/ms-marco-MiniLM-L-6-v2), and the optional LLM channel defaults to
`gpt-4o-mini` and never generates returned content. The position taken is that
a non-generative encoder is not a "model" in the sense the rule constrains,
since it cannot generate an answer; this is stated explicitly here so the
choice is disclosed rather than assumed. If the organisers read the rule more
narrowly, `CODEMEM_DENSE_ENABLED=false` and `CODEMEM_RERANK_ENABLED=false`
disable both channels entirely, and `CODEMEM_EMBED_BACKEND=openai` switches the
encoder to a hosted alternative — the entry stays compliant either way.

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
| SQLite / FTS5 | Storage, lexical index, per-user vector store | Public domain |
| PyYAML | Configuration parsing | MIT |
| pytest, httpx | Tests | MIT / BSD-3-Clause |
| `tiktoken` (optional) | Token accounting | MIT |
| `numpy` (optional, `dense` extra) | Vector arithmetic for dense retrieval | BSD-3-Clause |
| `sentence-transformers` (optional, `dense` extra) | Embedding and cross-encoder runtime | Apache-2.0 |
| `openai` (optional, `llm` extra) | Client for hosted encoder/LLM backends | Apache-2.0 |
| BAAI/bge-small-en-v1.5 (default `CODEMEM_EMBED_MODEL`) | Dense embeddings | MIT |
| cross-encoder/ms-marco-MiniLM-L-6-v2 (default `CODEMEM_RERANK_MODEL`) | Cross-encoder reranking | MIT |

The two default models are non-generative encoders distributed via Hugging Face
(`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`), and their
model cards carry the MIT licence. Both channels are runtime-switchable
(`CODEMEM_DENSE_ENABLED=false`, `CODEMEM_RERANK_ENABLED=false`), so the service
also runs fully model-free. No FAISS or other vector-index library is used:
vectors live in SQLite and are searched brute-force within one `user_id`.

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
