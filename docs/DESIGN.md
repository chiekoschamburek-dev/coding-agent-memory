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
   test. Three of those rules were tightened against the corpus rather than by
   reasoning (`scripts/audit_chunk_kinds.py` reports the distribution and the
   suspect labels):
   * a segment is a **diff** only if a diff header is present, not merely
     because half its lines start with `+`/`-`. Markdown bullets and numbered
     steps begin the same way, and 43% of the chunks labelled `diff` contained
     no hunk at all;
   * a segment is **config** only if it opens with a key, section, comment or
     brace and its values are short. An issue description with indented
     `problem_statement: …` lines otherwise reads as configuration — 95% of the
     `config` chunks were the task prompt, which also mis-set their `lang` to
     `yaml` and unlocked the aggressive symbol/command extractors on prose;
   * **test output inside an unlabeled fence** is recognised by content (at
     least half test-shaped lines plus one test-run marker). 99.2% of this
     corpus's fences carry no language tag, so the language-based branches could
     never reach it and every such block fell through to `code`.
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

**One session per request, but a session may span several requests.** The
contract's `session_id` is a single value, and which of the platform's messages
arrive together is its choice: the example `request_id`
(`eval:<run_id>:locomo_refined:conv-0:chunk-0`) carries a segment suffix, and the
`messages[].timestamp` note says "分段不改变其值和消息顺序" — segmentation is
expected. We support it (the `chunk` uniqueness key includes `request_id`, so
segments never overwrite each other), but two fields restart per request:
`msg_index` and `ord` both begin at 0 again, so segment 2's message 0 is
indistinguishable from segment 1's. Nothing in Search reads either field, so
retrieval is unaffected; provenance is.

**The dedup key has no `session_id` in it, and that is not free.** Measured on
the proxy corpus: 34 186 messages chunk into 37 115 chunks, of which 28 675 become
memories. Splitting the loss by cause:

| | chunks | share | is it a loss? |
|---|---|---|---|
| same session repeated itself | 3 235 | 8.7 % | no — the dedup is doing its job |
| **another session already had the text** | **5 205** | **14.0 %** | **yes — provenance is lost** |

`UNIQUE (user_id, sha)` cannot tell "the same session re-sent the same text" from
"two different sessions both said this", and the second case is 14 % of the
corpus. Identical text lands under whichever session wrote first.

**How much this actually costs.** Less than the 14 % suggests, for two reasons.
First, what gets dropped is repetition — stock assistant openings, identical tool
errors — not a session's substance: measured per session, the keep rate is 65 %
to 100 % (mean 76.3 %), and **not one of the 300 sessions loses more than a third
of its memories**, none is emptied. Second, because the drop condition is *exact
text equality*, the receiving session already contained that text, so dedup never
moves foreign content into another session's assembly group — it cannot corrupt a
group's context, only shorten a session's own list.

So for the returned payload the effect is small, and it is not a reason to make
the key three columns. It does make `memory.session_id` an unreliable record of
authorship, which is worth knowing before using it to reason about attribution —
see `eval/README.md` on why the session is a packing unit and a proxy label, not
a contract unit.

Whether that matters depends on the reader. For answer quality it is close to
harmless — the text is still in the pool and still retrievable. For **provenance
and for any session-level measurement** it is not: the memory a session
contributed can be attributed elsewhere, which is one of the reasons
`eval/run_benchmark.py` sees fewer retrievable entries per session than the
transcript contains messages. Making the key `(user_id, session_id, sha)` would
keep both properties; it is a schema change and has not been made.

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

Four channels, chosen because their failure modes differ:

| Channel | Signal | Fails when |
|---|---|---|
| `lexical` | FTS5 BM25 over text plus identifier sub-tokens | wording differs |
| `entity` | exact identifier equality, IDF-weighted | identifiers absent |
| `dense` | embedding similarity over stored vectors | — (optional, off by default) |
| `recency` | age decay over recent memories | query is time-independent |

A `sparse` column carries identifier sub-tokens (`read_token` ⇄ `readToken` ⇄
`read`), because the `unicode61` tokenizer keeps camel and snake case whole.

**Dense recall is off by default, on measurement.** With reranking enabled it
added no metric gain while making Add ~26× slower; it does help when reranking is
unavailable. See `eval/README.md` for the full ablation and the caveat that the
proxy's ground truth cannot credit the exact case dense is best at.

### Reranking

A cross-encoder scores each (question, memory) pair *jointly*, which is what lets
it separate the two cases the recall channels cannot:

```
"IndexError in tokenizer.py, guarded the empty buffer"  -> relevant
"reformatted tokenizer.py for the linter"               -> not relevant
```

Both share every identifier, so BM25 and embeddings see them as near-identical;
only joint reading distinguishes them. This is the largest measured gain of any
stage and it runs only at search time, so It costs Add nothing and is enabled by
default.

Two implementation details were forced by measurement:

- **Documents are clipped by tokenizer, in one batch.** Characters are a poor
  cost proxy: 2048 characters can be 630 tokens while 40 characters is 21, and
  latency scales with real tokens (4 ms/doc at 21, 48 ms/doc at 1034). Token
  clipping took long-memory reranking from 48 to ~14 ms/doc, and calling the
  tokenizer once for the whole pool rather than once per document took a
  120-document pool from 6.0 s to 2.1 s.
- **Scores use a fixed temperature, never max-normalisation.** Normalising by the
  head's maximum made each memory's contribution depend on which other memories
  happened to be reranked, so results changed with pool size. The measured
  sequence was incoherent (MRR 0.818 → 0.772 → 0.684 as the pool grew); with a
  fixed temperature it is monotone (0.684 → 0.746 → 0.772).

### Listwise reranking (optional, off by default)

A final stage where an LLM reads the candidate list together and scores it. Its
advantage over the cross-encoder is *comparative* judgement: a per-pair scorer
cannot see that forty candidates all come from the same repository, so it cannot
know that "this one is the explanation" is the discriminating fact.

**Generation boundary.** This stage returns numbers only. The model is asked for
a JSON array of relevance integers, which is parsed into floats and blended into
the existing score. No model output reaches ``data[].content``, which remains a
verbatim span of stored memory text. Scoring existing memories is not generation,
so rule 1 is respected; ``tests/test_listwise.py`` asserts that even a reply
containing prose contributes scores and never text.

**Robustness.** A missing endpoint, an HTTP failure, an unparseable reply, or a
wrong-length array leaves the previous ranking untouched. A partial answer is
discarded outright rather than padded, because mixing judged and invented scores
would corrupt the ranking invisibly. Candidates beyond the judged head keep a
neutral score rather than being treated as irrelevant.

**Why it is off.** On the proxy benchmark it lost at every setting while costing
~5x search latency, and attenuating its weight drifted results monotonically back
to the baseline — the signature of noise rather than signal. The caveat is that
the proxy's ground truth (file overlap) measures something different from what a
judge optimises (usefulness), so the stage is unproven here rather than refuted.
See ``eval/README.md`` for the numbers and the hand-inspected counterexample.

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

#### The returned payload

The contract fixes the shape, and the example value for ``content`` is plain
remembered text:

```json
{"data": [{"id": "mem_123", "content": "remembered fact text",
           "score": 0.87, "created_at": "2026-07-01T12:00:00Z"}]}
```

So ``content`` is a **verbatim span of stored memory text** — no header, no
labels, no rewriting — and ``created_at`` carries the **source timestamp** the
platform supplied, falling back to our persistence time only when the source had
none. An earlier revision put a ``[memory]/[file_path]/[time]`` header inside
``content`` and left ``created_at`` as our own write clock; that wasted the field
the contract provides and put information in the text that an auditor could not
find in the Add input.

What we do *not* put in ``content``, and why:

- **Extracted identifiers.** They are retrieval keys, matched against the index
  during ranking. Duplicating them in the payload spends answer-model budget on
  text that is already implied by the content.
- **A compact "summary" of the memory.** The contract says returned content is
  "preserved verbatim for audit", which only means something if an auditor can
  locate our output in the input. A summary asserting something the trajectory
  never said would be fabricated evidence rather than a digest of it. A real
  session trajectory is also usually already more compact than a lossy
  re-rendering of it, so paraphrase buys little and risks a great deal.
- **The superseded marker.** Whether a memory was superseded is our judgement, not
  memory text, so it travels on the ``id`` (which the contract only requires to be
  a stable identifier) rather than editing the content.

Truncation is the one permitted modification, and it is **selection rather than
rewriting**. The span is chosen by query-term density rather than by taking the
opening lines, because in a long trajectory the framing sits at the top and the
diagnosis sits in the middle; elision is marked with ``…`` so a reader can tell a
span from a whole.

`tests/test_traceability.py` enforces all of this, and fails if free-form
generation is ever introduced into returned content.

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
