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

Every retrieved item is a row here. `kind` is `chunk` today, with `card` defined
below and `episode` still reserved. Keeping generation confined to Add is
enforced structurally: Search cannot return text that was not first written to
this table.

#### Card schema

A card is one comparable object per topic span inside one session, so a session
enters ranking as a single candidate rather than as ~100 siblings competing for
the same slots. It is *addressable*, not *quotable*:

```
memory (kind = 'card')
  id            card_<session_id>_<ordinal>
  user_id       the card's session's owner; filtered exactly like a chunk
  session_id    never spans sessions
  span_first    row id of the first covered chunk   \ same basis as
  span_last     row id of the last covered chunk    / Store.session_span
  entities      L2 identifier set over the span, IDF-weighted
  overview      LLM text, read only by the scorer
  content       NULL
  created_at    source timestamp of the earliest covered message
  superseded_by as for any memory row
```

`span_first`/`span_last` bound the card against `session_span`, which is computed
for the position tilt anyway; the card adds no new ordering primitive. Note this
is a topic span across messages, unrelated to `chunker.segment()`, which splits
one message body at structural boundaries.

The invariants that make it safe, each one testable:

1. **`content IS NULL`, and `data[].id` never resolves to a card.** A card is
   expanded to the chunks it covers before assembly, so everything the platform
   sees is still a verbatim span of Add input. `tests/test_traceability.py`
   rejects generated text in `content`; a card returned whole would fail it.
2. **Span integrity.** Card spans within a session are non-overlapping and each
   bound names a real chunk row. This is what makes "expand then return" total:
   a card can never point at text that does not exist.
3. **`overview` reaches no prompt and no payload.** Its only reader is scoring,
   which emits a number. This keeps the single documented exception — an LLM in
   the Search path scoring existing memories, never generating returned text —
   the only exception.
4. **Determinism.** Card identity and spans are pure functions of Add input plus
   configuration, and `overview` is cached by the content hash of its span
   (the existing enrichment cache), so a re-Add or a cache hit reproduces the
   same row and the same ranking. A non-deterministic scorer input would put the
   platform's reproduction check at the mercy of query order.
5. **A card qualifies nothing.** Eligibility still requires a lexical or entity
   hit on an actual chunk, as with recency and dense. Without this, an
   approximate channel becomes a way to walk a candidate past the noise gate.
6. **`created_at` carries source time**, never the write clock — the card's
   processing date is a fact about our pipeline that no auditor can find in the
   input.

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

One pass over ten stages. Six are always on; four are optional and **ship
switched off**, each for a measured reason recorded below. Two invariants hold
across all ten: nothing here generates text (§1, and
`tests/test_traceability.py` fails if it ever does), and every read is scoped by
`user_id`.

### Stage order

```
 1  plan_query            question + options -> probes, entities, intent
        |
 2  _hyde_probe           [OFF] one LLM call rewrites the question as the note a
        |                        past session would have recorded -> extra probe
 3  retriever.recall      lexical | entity | dense | recency  --(RRF k=60)-->  pool 300
        |
 4  score_candidates      0.40 RRF + 0.15 coverage + 0.30 strength + 0.15 id
        |                        (+ optional 5th "operative" term), normalised to 1.0
 5  _rerank               [ON]  cross-encoder over the head (120), blended at w=0.65
        |
 6  _listwise_rerank      [OFF] LLM comparative scoring over the head (40)
        |
 7  _session_features     [OFF] F1 + F2 rank-fused -> session block order
 8  _llm_select_sessions  [OFF] top-8 session digests -> 2 picks, 10 s hard budget
        |                        (takes precedence over 7 when it succeeds)
 9  evidence.assemble     session-major slots -> gate -> verbatim spans -> data[]
        |
10  _dense_fill           [OFF] append-only: at most 4 dense-only memories at the tail
```

Stages 7 and 8 reorder **sessions**, not chunks: they permute session blocks and
hand off to `assemble`, which still applies the noise gate, the per-session cap
and the token budget unchanged. That separation is deliberate — an ordering lever
may promote a session, but it may not smuggle an item past a gate.

### Query planning

The question and the option labels (when present) produce retrieval probes:
identifiers are extracted verbatim and intent is classified as debug or develop.
Intent is detected partly from `exception` entities, because a word-boundary
search for "error" does not match camel-case `IndexError`.

**On `options`.** Options are sent without gold answers. They are used *only* to
widen what the question asks about — often the option wording names the relevant
subsystem, which is useful when the question itself is vague. We never select
memory per option ("which memory supports option B"), because that shades into
disguising an answer as memory, which the rules forbid.

**HyDE probe (off).** `hyde_probe` makes one relay call that rewrites the raw
issue as a 1–3 sentence first-person note naming concrete identifiers and stating
a cause or fix, then appends it to `plan.probes`. It exists because of a measured
representation gap: questions arrive as raw issues while answers are stored as
recorded cause-prose, and the shipped plan seated the answer session in the top-8
for only 22 of 70 questions on the claim anchor while 36 more were admissible at
rank 9+. Offline replay moved that menu to 29/70 (net +7, 9 wins / 2 losses). The
probe rides the lexical and dense channels only — the entity channel is untouched,
so any effect attributes to those two. No relay, a timeout, or an empty reply
returns the plan unchanged. It stays off pending the end-to-end verdict.

### Recall

Four channels, chosen because their failure modes differ:

| Channel | Signal | Fails when |
|---|---|---|
| `lexical` | FTS5 BM25 over text plus identifier sub-tokens | wording differs |
| `entity` | exact identifier equality, IDF-weighted | identifiers absent |
| `dense` | embedding similarity over stored vectors | — (see admission below) |
| `recency` | age decay over recent memories | query is time-independent |

A `sparse` column carries identifier sub-tokens (`read_token` ⇄ `readToken` ⇄
`read`), because the `unicode61` tokenizer keeps camel and snake case whole.

Fusion is Reciprocal Rank Fusion (k=60) with per-channel weights — `lexical` 1.0,
`entity` 1.15, `dense` 0.70, `recency` 0.08. Two properties of RRF had to be
corrected for, both found by measuring at realistic corpus size: recency must be a
near-tie-breaker rather than a peer (a peer-weighted recency channel contributed
more than a five-fold BM25 win, and the relevant memory slid from rank 1 to 5),
and rank fusion alone discards match strength (see the `strength` term below).

Two knobs shape the pool, both off by default and both motivated by the same
finding — a session holds ~96 entries on the proxy corpus, so an entry-counted
pool is dominated by whichever few sessions matched first:

- `recall_channel_depth` — pull a deeper slice per channel (default: reuse
  `recall_per_channel`, 120);
- `candidate_per_session` — cap entries per session inside the pool before it is
  truncated to `candidate_pool` (300). Measured: an uncapped pool of 300 spanned
  ~3 sessions and `rerank_top_n` 120 barely 1.3.

### Scoring

`0.40·RRF + 0.15·coverage + 0.30·strength + 0.15·identifier`, times an
intent/kind bonus and a superseded penalty, then normalised so the head is 1.0.

Four deliberate choices:

- **Coverage counts only informative channels.** Recency is excluded. Being new is
  not evidence of relevance, and counting it would let an irrelevant memory clear
  the gate on freshness alone.
- **An absolute evidence requirement.** Scores are normalised against the best
  candidate, so a relative threshold could never drop anything. Eligibility
  therefore requires that at least one informative channel (lexical or entity)
  actually found the memory. Without this, every query returns an irrelevant item;
  with it, an unrelated question correctly returns `[]`.
- **Stopwords are removed before FTS matching.** FTS terms are OR-combined, so
  leaving `the`, `of` or `is` in the query makes *every* memory match *every*
  question, which silently disables the noise gate.
- **Damped identifier signal.** Identifier evidence is raised to the 1.5 power
  after normalisation, so one strong match helps but a memory cannot win on
  identifiers alone — same-repo distractors share paths too.

Two optional extensions, both off:

- **`operative_rank_weight`** adds a fifth term, "this chunk records an action"
  (`[tool Edit]`, `has been updated`, a diff header — bounded at 3 markers so a
  ten-hunk diff is not an outlier, and gated on the identifier signal so a chunk
  cannot be promoted for containing an edit to an irrelevant file). Setting it
  rescales the other four terms so the total stays on the 0..1 scale the noise
  gate is calibrated against. Off by default: what it targets is a property of
  *which chunk* carries a session's score, and that chunk names a file the task
  touched only 12.9% of the time — a ranking term cannot fix a representation
  problem, and measured worse as a kind-level bonus (§7 of `eval/README.md`).
- **`dense_eligible`** admits candidates that *only* the dense channel found,
  behind an absolute similarity floor (`dense_eligible_min_similarity`, 0.45) and
  an optional count cap. Their scores are rescaled within the band above the floor
  rather than compared to BM25 magnitudes. Measured net zero on the B anchor, so
  the cheaper `dense_fill` variant is the one to try first.

### Reranking

A cross-encoder scores each (question, memory) pair *jointly*, which is what lets
it separate the two cases the recall channels cannot:

```
"IndexError in tokenizer.py, guarded the empty buffer"  -> relevant
"reformatted tokenizer.py for the linter"               -> not relevant
```

Both share every identifier, so BM25 and embeddings see them as near-identical;
only joint reading distinguishes them. This is the largest measured gain of any
stage (MRR +0.093, nDCG@10 +0.067) and it runs only at search time, so it costs
Add nothing and is on by default.

Three details, each forced by measurement:

- **The window is chosen, not truncated from the front.** `rerank_span_tokens`
  (0 = off) feeds the model the same query-term-density-with-operative-weight
  span that `assemble` would return, rather than a character prefix. A 2 048-
  character prefix can be 630 tokens while 40 characters is 21, and latency
  scales with real tokens — token clipping took long-memory reranking from 48 to
  ~14 ms/doc, and one batched tokenizer call took a 120-doc pool from 6.0 s to
  2.1 s.
- **Scores use a fixed temperature, never max-normalisation.** Normalising by the
  head's maximum made each memory's contribution depend on which other memories
  happened to be reranked, so results moved non-monotonically with pool size.
  `rerank_probability_scores` handles the other model family: a bge-reranker
  already emits 0..1 and pushing it through the sigmoid flattens every candidate
  toward 0.5.
- **`rerank_session_level`** (off) scores session summaries instead of chunks,
  which moves the cross-encoder from the chunk ranking to the session ranking
  that stages 7–9 consume. Measured as part of the session-major work; not the
  shipped path.

Blend: `final = 0.35·fused + 0.65·rerank`. Weight 0.65 was swept (0.85 degrades,
0.45 is measurably worse). When the model is unavailable the stage is skipped and
the fused recall ranking stands.

### Optional LLM ordering (stages 6–8)

`tests/test_listwise.py` pins the rules these stages share: they return numbers or
a permutation, never text; every failure mode (no endpoint, HTTP error,
unparseable or wrong-length reply, timeout) leaves the previous order untouched;
and none of them may re-open a decision the noise gate made.

**Listwise chunk scoring (off).** An LLM reads the head of the candidate list and
scores each memory comparatively. It lost at every setting on the proxy while
costing ~5× search latency, and attenuating its weight drifted results
monotonically back to the baseline — the signature of noise, not signal.

**Session-feature fusion (off).** Two session-level signals the per-chunk view
cannot produce, rank-fused with the head order at weights 2.0/2.0/2.0:

- **F1** — cosine between the query and the session's **first raw message**. The
  issue statement lives at the trajectory head and is written in the vocabulary
  queries arrive in, whereas the chunks that carry the session's score are in tool
  vocabulary.
- **F2** — how much of the query's *rare* vocabulary (terms appearing in at most
  two candidate sessions) the session's pooled chunks cover **as a union**, which
  a per-chunk max structurally cannot see.

Offline replay: macro 0.4746 → 0.535. Kept off until it is judged on all three
instruments, since it costs one embed call per search.

**Session selection (off).** The strongest-looking and most carefully distrusted
stage. The top ~8 candidate sessions are rendered into compact digests (file
names, opening 280 chars, two longest chunks) and one `gpt-4o-mini` call picks the
two that record the cause or the fix; those two blocks go first. It makes the same
judgement the platform's answer model makes ("can this context answer this
question") but over session summaries rather than single chunks — which is exactly
what the failed listwise stage got wrong.

It ships off because of a cross-anchor reversal: it was the seven-arm champion on
the file-overlap instrument (+5.2 pp over the deterministic ordering) and then
**reversed** on the leak-free claim-topical instrument — 0.229 against 0.300 for
the shipped ordering, which puts it **below the 0.243 no-memory floor**, the
cardinal sin under the RAG doctrine. The mechanism is confirmed by a
one-call probe: the selection judgement inherits the answer model's prior. On the
old instrument that prior pointed at the gold (its floor was 0.328 because the
options leaked), so aligning with it won; the claim instrument is built
adversarially so the prior points at distractors, and alignment amplifies the
error — 27 of 140 picks landed on the question's own distractor sessions. Under
cross-anchor uncertainty the deterministic ordering, which is never below the
floor on either instrument, is the default. One env flip re-enables it.

Supporting knobs: a 10-second hard budget (the relay answered in ~1 s when
healthy, so the fallback fires well inside the 30-minute ceiling — quantify the
fallback rate during deployment soak), and `session_select_votes`, which is
**documented dead**: votes=3 scored *below* the one-shot (0.214 vs 0.229), because
majority voting fixes variance, not bias — when a judgement is systematically
attracted to distractors, repeated calls agree with each other and the majority
entrenches the error. The earlier "8 of 11 misses flip on re-call" was boundary
jitter measured on one instrument only.

### Noise gate and evidence assembly

`assemble` treats **sessions as the ranking unit and chunks as the evidence
unit**. Candidates arrive in descending score order and are grouped by session, so
each session's head is its best chunk and the group order is the session order.

Per session, in order:

1. **Noise gate.** A session whose head scores below `min_evidence_score` (0.15)
   ends the loop rather than padding the answer model's prefix — and the gate
   reads the group *before* any intra-session reordering, so a session's score
   stays its strongest evidence whatever the levers below decide to show first.
2. **Position prior** (`evidence_position_weight`, on at 1.0) — within an already
   chosen session, prefer later chunks: trajectories read a file before changing
   it, so the edit sits late (the chunk naming a patched file sits at relative
   position 0.575, and the edits among them at 0.661, against 0.500 overall).
3. **Operative promotion** (`evidence_operative_promotion`, 2) — for the top two
   sessions only, move the chunk that records an action into the first slot. This
   is a double-edged lever: applied to *every* session it recovered decisive
   evidence but also surfaced other sessions' edits, which is what makes a
   distractor look modified too (ambiguity 0.567).
4. **Per-session cap** (`max_evidence_per_session`, 5) and a payload-wide session
   cap (`evidence_max_sessions`, 2): with fewer sessions competing for the token
   prefix, distractor operative chunks stay out while the answer session gets
   enough slots to expose its decisive evidence.
5. **Selection of the text.** Each item is a verbatim, line-aligned span. When the
   operative weight applies — only for the top-ranked session — selection is
   **confidence-aware multi-span**: extract every operative block (an operative
   line plus a little context) first, then fill the remaining budget with the
   densest non-operative window. A single continuous window can cover one edit and
   drop another in the same session. Later sessions keep single-window selection so
   distractor edit markers do not inflate ambiguity.

Cards are never returned. `memory.kind == 'card'` is skipped at every emission
path (assembly, promotion, dense fill): a card's text is generated, and
`data[].content` must stay a verbatim span of Add input. A card may still carry
its session's rank, because it is a row in `memory` like any other. With
`card_expansion` on, a session whose *only* admitted member is its card may have
that card vouch its own chunks into the payload — verbatim spans, position-
prioritised, capped, deduplicated and budgeted like any other slot fill. The card
qualifies nothing on its own: a session with no chunk of its own that cleared the
informative-channel gate is not evidence.

Budget and shape: the first `evidence_full_count` (8) items render at up to
`evidence_item_tokens` (800) tokens and the tail at `evidence_ptr_tokens` (110),
inside `evidence_budget_tokens` (60 000) — well under the platform's 117 760 input
window. Iteration stops rather than truncating the strongest evidence. Scores
decay by position (`-1e-5` per step, clamped at `1e-6`) so the returned order and
the returned scores can never disagree, which the rounding to six decimals makes
non-trivial.

**Dense fill (off).** After assembly, at most `dense_fill_max` (4) memories that
*only* the dense channel reached may be appended to the tail, each
behind `dense_fill_min_similarity` (0.50) and rendered at 110 tokens. Three
exclusions define it as a fill for the lexical blind spot rather than a second
chance: not already returned, not from a session already represented, and **not
contested by lexical or entity** — something the lexical channels saw and left
behind stays left behind. It never fires on an empty result, because the noise
gate abstaining is a decision about the query. Scores continue strictly
decreasing.

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
rewriting**: elision is marked with ``…`` so a reader can tell a span from a whole.

`tests/test_traceability.py` enforces all of this, and fails if free-form
generation is ever introduced into returned content.

### What Search may not do

Collected because each of these was proposed at some point and rejected on
evidence, not taste:

- generate or rewrite any returned text — the LLM stages produce numbers and
  permutations only;
- select memory per option, or otherwise let the answer leak into ordering;
- let recency, a card, or an elevated session score qualify an item that no
  informative channel found;
- reorder past a gate: stages 7 and 8 choose *which session first*, never *what
  is admissible*;
- cross `user_id`, including in the entity, recency and dense channels, all of
  which are query-independent and therefore easy to get wrong silently.


## 5. Determinism and reproducibility

Chunking, identifier extraction, and scoring are all pure functions of stored
data plus configuration. No randomness, no time-dependent behaviour except the
documented recency decay. The same corpus and configuration produce the same
ranking, which is what makes the platform's reproduction check safe and our own
ablation results meaningful.

That guarantee covers the **shipped configuration**, and it is worth stating
where the boundary now sits, because Search gained stages that cross it:

- the deterministic core (plan, recall, score, cross-encoder rerank, assemble,
  dense fill) is reproducible: an accidental unflagged rerun of the proxy
  benchmark reproduced every metric to the digit (`eval/README.md`), and
  `scripts/diagnose_determinism.py` exists to localise any divergence that does
  appear by hashing the evidence payload across runs and across processes;
- the optional LLM stages (`hyde_probe`, `listwise_enabled`, `session_select_llm`)
  call an external endpoint, and that endpoint was measured to be
  **non-deterministic at `temperature=0`** (the same prompt returned "B" four
  times and "C" four times in eight calls; an explicit seed did not stabilise it).
  Enabling them makes Search reproducible only up to the ordering they choose, and
  `session_select_votes` > 1 samples that variance rather than removing it;
- which stages are active is configuration, so the reproduction check must record
  the env, not just the image digest. The default deployment keeps every LLM stage
  off, which is a determinism argument as much as a quality one.

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
