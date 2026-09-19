# Design

## 1. The problem shape

We receive engineering trajectories from a repository and must later return the
memory that helps answer a question about that repository. Two properties of the
setup determine everything below.

**We only ever see trajectory text.** There is no repository checkout and no
source access. Every identifier we index — a path, a symbol, an exception class
— is a *mention* extracted from text, never a fact parsed from real source.
This is also why we never infer "the current state of the repository": the
memory holds what happened and when, not what the code looks like now.

**The noise is same-repository.** Distractors share vocabulary, directory
layout, package names, and style with the relevant evidence. Lexical and
embedding similarity are therefore systematically inflated across the whole
pool, which is why exact identifier matching and an explicit noise gate are
load-bearing rather than optional refinements.

The platform feeds the answer model a token-counted prefix of our ranked `data`,
so output order is effectively the score, and low-value items actively harm the
result by consuming that prefix.

## 2. Data model

All layers are keyed by `user_id`, the sole isolation boundary.

### L0 — `raw_message` (lossless, immutable)

We receive each trajectory once. The raw text is stored verbatim with its
`request_id`, `session_id`, role, timestamp and content hash. This is the basis
for provenance, for audit, and for re-chunking or re-enriching later without
re-requesting data. Nothing downstream is allowed to be the only copy of
anything.

### L1 — `chunk` (deterministic)

Structure-aware pieces of a message. Segmentation runs in this order:

1. split on marked code fences (` ``` `, `~~~`) — fence boundaries are hard;
2. inside a fence, classify by *content*, not by the fence language label
   (a `diff` inside a `python` fence is still a diff);
3. outside fences, group blank-line-separated paragraphs, then split again so a
   `diff --git` header starts its own segment;
4. classify each segment as prose, code, diff, log, stacktrace, cmd, config, or
   test;
5. pack prose paragraphs up to a token target; keep structural segments whole;
6. only split on line boundaries when a segment exceeds the cap, and hard-cut by
   characters solely for a single pathological line.

Chunking is a pure function of the input, so identical text always yields
identical chunks — a precondition for reproducible scores and safe reindexing.

### L2 — `chunk_entity` (deterministic, no LLM)

Identifiers extracted by regex: `file_path`, `file_name`, `dir`, `symbol`,
`exception`, `test`, `cmd`, `pkg`, `issue_id`, `lang`.

Guards that matter in practice: version strings (`3.11.2`) are not file paths;
`UTF-8` and `HTTP-404` are not issue references; prose mentioning `foo.py` does
not make the chunk Python. Identifiers are de-duplicated per chunk and capped
per type so a large blob cannot flood the index.

### L3 — `memory` (the only table Search reads)

Every retrieved item is a row here. `kind` is `chunk` today, with `card` and
`episode` reserved for enrichment. Keeping generation confined to Add is
enforced structurally: Search cannot return text that was not first written to
this table.

### Governance

- `entity_timeline` — entity to time-ordered memory, so recency can *weight*
  rather than filter.
- `repo_profile` — per-`user_id` document frequency per identifier. Serves as an
  IDF table, as a soft repository profile, and as a diagnostic: a bimodal
  profile means one `user_id` spans several repositories, in which case the
  profile should partition candidates rather than be trusted.
- `supersedes` / `memory.superseded_by` — soft forgetting. A newer memory may
  down-weight an older one, never delete it: an old fix that a later session
  replaced is still a plausible precedent, and we cannot verify the
  repository's current state. **Scaffolded but not yet active**: the table, the
  `superseded_by` column, the 0.65 scoring penalty, and
  `Store.mark_superseded()` all exist, but nothing populates the links yet.
  Deciding when a later session genuinely invalidates an earlier one needs
  calibration against the benchmark, and an uncalibrated heuristic would quietly
  down-weight valid evidence — so it stays off until `eval/` can measure it.
- `llm_cache` — content-hash keyed, so retries and repeated text never pay twice.

## 3. Add

1. Validate the payload; enforce size limits.
2. Check `request_id`. A retry is answered from the recorded outcome and writes
   nothing, so retries stay idempotent as the contract requires.
3. Chunk and extract identifiers (both deterministic).
4. Persist raw messages, chunks, entities and memories in one transaction. The
   FTS index is maintained by trigger, so on commit the memory is durable *and*
   searchable — the precondition for `success: true`.
5. Enrichment (when enabled) runs afterwards and may not affect the outcome.

**Degradation.** An internal deadline (25 min, under the 30-min contract limit)
guards enrichment. If it is missed or the LLM fails, the response is still
`success: true` with the raw memory searchable. Add cannot be failed by a
quality-improving step.

A second request carrying identical text under a different `request_id` maps to
the same memory via `UNIQUE (user_id, sha)`, so re-sends cannot duplicate the
pool.

## 4. Search

### Query planning

The question and the option labels (when present) produce retrieval probes:
identifiers are extracted verbatim, and intent is classified as debug or
develop. Intent is detected partly from `exception` entities, because a
word-boundary search for "error" does not match camel-case `IndexError`.

**On `options`.** Options are sent without gold answers. They are used *only* to
widen what the question asks about — often the option wording names the relevant
subsystem, which is useful when the question itself is vague. We never select
memory per option ("which memory supports option B"), because that shades into
disguising an answer as memory, which the rules forbid.

### Recall

Three channels, chosen because their failure modes differ:

| Channel | Signal | Fails when |
|---|---|---|
| `lexical` | FTS5 BM25 over text plus identifier sub-tokens | wording differs |
| `entity` | exact identifier equality, IDF-weighted | identifiers absent |
| `recency` | age decay over recent memories | query is time-independent |

A `sparse` column carries identifier sub-tokens (`read_token` ⇄ `readToken` ⇄
`read`), because the `unicode61` tokenizer keeps camel and snake case whole.

### Fusion

Reciprocal Rank Fusion (k=60) with per-channel weights. RRF needs no
cross-channel score calibration, which matters because BM25 and identifier
weights are on incomparable scales.

Two properties of RRF had to be corrected for, and both were found by measuring
at realistic corpus size rather than by reasoning:

**Recency must be a near-tie-breaker, not a peer.** RRF compresses magnitude into
rank distance, so a peer-weighted recency channel contributed more to the fused
score than a five-fold BM25 win did. With hundreds of newer same-repository
distractors, the genuinely relevant memory slid from rank 1 to rank 5. Recency's
weight is therefore 0.08, not 0.35.

**Rank fusion alone loses match strength.** Because RRF discards magnitude, a
memory that wins BM25 by 5x (13.98 vs 2.77 in a measured case) scored almost the
same as one that barely cleared the threshold. The final score therefore blends a
*strength* term — the best per-channel score, normalized per query — alongside
the fused rank.

### Scoring

`0.40·RRF + 0.15·coverage + 0.30·strength + 0.15·identifier`, times an
intent/kind bonus and a superseded penalty.

Four deliberate choices:

- **Coverage counts only informative channels.** Recency is excluded. Being new
  is not evidence of relevance, and counting it would let an irrelevant memory
  clear the gate on freshness alone.
- **An absolute evidence requirement.** Scores are normalized against the best
  candidate, so the top item is always 1.0 and a relative threshold could never
  drop anything. Eligibility therefore requires that at least one informative
  channel actually found the memory. Without this, every query returns an
  irrelevant item; with it, an unrelated question correctly returns `[]`.
- **Stopwords are removed before FTS matching.** FTS terms are OR-combined, so
  leaving `the`, `of` or `is` in the query makes *every* memory match *every*
  question. This silently disabled the noise gate: queries like "How do I bake
  sourdough bread?" returned memories. Verified in `tests/test_ranking_scale.py`.
- **Damped identifier signal.** Identifier evidence is raised to the 1.5 power
  after normalization, so one strong match helps but a memory cannot win on
  identifiers alone — same-repo distractors share paths too.

### Noise gate and evidence assembly

Items below `min_evidence_score` are dropped, and iteration stops rather than
padding the prefix with distractors.

Each item is assembled **only** from stored content: a fixed header built from
deterministically extracted identifiers, then a verbatim excerpt. The header is
formatting, not new claims.

Items form a pyramid: the head is rendered in full form and the tail in a
compact pointer form, so the token budget buys coverage without truncating the
strongest evidence. The score sequence is forced strictly decreasing so returned
order and returned scores can never disagree.

**A per-session cap is required, not cosmetic.** One session produces many chunks,
and without a cap they crowd out other sessions: measured on the proxy benchmark,
~100 returned chunks collapsed to ~23 distinct sessions. Since a task is answered
from a session rather than from one chunk of it, recall depends on session
diversity. Capping at 3 per session raised recall@100 from 0.845 to 0.919 and
nDCG@100 from 0.681 to 0.705. Capped-out items are skipped, not truncated, so the
slot passes to the next session instead of shortening the list.

## 5. Determinism and reproducibility

Chunking, identifier extraction, and scoring are all pure functions of stored
data plus configuration. No randomness, no time-dependent behaviour except the
documented recency decay, and no LLM in the P1 path. The same corpus and
configuration produce the same ranking, which is what makes the platform's
reproduction check safe and our own ablation results meaningful.

## 6. Concurrency

SQLite in WAL mode with a serialized writer and a small read pool. Concurrent
Add retries of one `request_id` write once; interleaved users never observe each
other; reads proceed during writes. Verified in `tests/test_concurrency.py`.

## 7. Retention

Evaluation data and derived copies must be deleted within 30 days of task
completion and may not be used for training, fine-tuning, analysis, dataset
reconstruction, or redistribution. Accordingly: request bodies are not logged
(identifiers, sizes, timings only), the database holds no external copies, and
`DELETE /admin/users/{user_id}` erases every trace including index entries.
