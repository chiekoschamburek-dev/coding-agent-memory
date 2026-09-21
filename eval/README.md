# Evaluation

Two layers, kept separate on purpose.

## 1. Contract conformance (`pytest`)

The scored platform validates the API contract before it validates retrieval
quality, so the contract is tested directly: identifier echo, ``success`` only
after durability, always-present ``data``, the ``top_k`` ceiling, the error
envelope, authentication, isolation, idempotency, and concurrency.

```bash
pytest -q
```

## Scope: what we are graded on

The rules assign **Add and Search to us** and **Answer, Eval and publication to the
platform**, which uses its own locked answer model and prompt. The submission is
four things, all on our side of that line:

1. **recall** — find the prior sessions that matter;
2. **denoise** — drop what does not, rather than padding the token prefix;
3. **rank** — order by usefulness, since the platform consumes a token-counted
   prefix in our order;
4. **content selection** — decide what text each returned item carries, because
   the contract says ``data[].content`` enters Answer in the returned order.

Point 4 is easy to overlook and it is where the largest measured loss sits. It is
also entirely ours, and — unlike answer accuracy — **fully measurable without any
model in the loop**.

### Two things that are NOT ours, and why they were dropped from tuning

Measuring answer accuracy locally evaluates the platform's half with a model that
is not the platform's. It is also unreliable: on our relay, one prompt at
``temperature=0`` returned "B" four times and "C" four times out of eight, and an
explicit ``seed`` did not stabilise it (6/2 and 2/6 across variants). A single-pass
accuracy number from such an endpoint is a coin flip.

Those numbers are retained in this document only as case studies. **All tuning is
done against deterministic metrics**, which either move or do not.

## 2. Proxy retrieval benchmark (`eval/`)

**CAMBench Coding — the scored suite — is not public.** We therefore build a
proxy from [SWEContextBench](https://github.com/jiayuanz3/SWEContextBench), which
reproduces the two properties the Coding track is described as testing:

- **same-repository noise:** its 300 "Lite Past Experience" sessions span the
  *same 12 repositories* as its 99 Lite tasks, so distractors share vocabulary,
  file paths, and style with the relevant evidence;
- **reusing engineering experience:** memory items are real agent session
  trajectories (tool calls, diffs, test runs), not synthetic documents.

### Ground truth

SWEContextBench ships **no** task-to-session relevance mapping — verified: the
past-experience instances and the Lite instances have zero ``instance_id``
overlap, and no mapping file exists in the repository. We therefore define
relevance ourselves, in the most objective form the data supports:

> A past session is relevant to a task if it touched at least one file that the
> task's gold patch or test patch touches (normalized repository-relative paths).

This is code-grounded: it comes from the actual diffs and the actual tool calls,
not from a model's opinion. It is a **lower bound** on true relevance — a session
that would help for another reason (a shared technique, an architectural
decision) is not credited. The caveat is stored in the benchmark metadata so
results are never presented as official.

The two sides of the label come from different evidence, which is worth knowing
before reading a recall figure:

| side | where "files" comes from | what it means |
|---|---|---|
| task | `patch` ∪ `test_patch`, parsed from `+++ b/` and `diff --git` headers | the files the fix actually edits |
| session | the `"file_path": "..."` field in the transcript's tool calls | the files the agent **looked at** — read or edited alike |

Both sides are normalised to a repository-relative key (`testbed/` and
`owner__repo/` prefixes stripped), because the transcripts contain three
spellings of the same path. The asymmetry stands though: a session that only
*read* a file counts exactly as much as one that rewrote it, so "touched" is a
weaker signal than "worked on".

**How much of the label is structural.** A session touches a median of 6 files
(mean 6.6); a task's gold+test patch touches a median of 3 (mean 4.4). With
pools that size, 4.94 % of all same-repository (task, session) pairs overlap.
File overlap is a *common* event inside one repository, not a rare signal — which
is how a django task ends up with 5.86 relevant sessions while only 1.67 of them
share more than one file. That is the mechanism behind the 70 % single-file
figure in the decomposition section below.

The label is also **binary and session-level**: no relevance strength, and no
per-entry judgement, even though the platform scores memory entries.

**We use SWEContextBench's data, not its metric.** Its own evaluation is
SWE-bench's: you submit `{instance_id}_preds.json` holding a `model_patch`,
`evaluation.sh` runs `swebench_memory.harness.run_evaluation` in Docker, and an
instance counts as solved when its `FAIL_TO_PASS` tests pass and its
`PASS_TO_PASS` tests still pass — i.e. **resolve rate**. That measures whether an
agent can *repair* the bug using past experience. It says nothing about
retrieval, and no number in this file is comparable to it.

So the relationship is three layers deep, and each step is an approximation:

| layer | question | metric | defined by |
|---|---|---|---|
| SWEContextBench official | can the agent fix the bug using past experience | resolve rate (FAIL_TO_PASS / PASS_TO_PASS) | upstream, SWE-bench harness |
| our proxy | did retrieval surface the right sessions | MRR / recall@k / nDCG | us, file-overlap gold |
| CAMBench Coding | does the answer model answer correctly | not public; our proxy for it is accuracy | the platform |

Our relevance definition is a proxy for the upstream notion of "related", which
the paper derives from dependency and reference relationships between GitHub
issues and PRs — a signal the dataset does not expose as a per-pair mapping.
File overlap is our substitute for it, and the 70 % single-hot-file result in
the decomposition section below is how loose that substitute is.

### Running it

```bash
# one-time: clone the dataset (151 MB)
git clone --depth 1 https://github.com/jiayuanz3/SWEContextBench benchmark/SWEContextBench

python eval/build_benchmark.py --out eval/data/benchmark.json
python eval/run_benchmark.py --data eval/data/benchmark.json \
    --out eval/results/baseline.json --dump-per-query eval/results/per_query.json
```

The runner goes through the real HTTP contract, so schema, auth and ``top_k``
handling are exercised. Returned memories are mapped back to their source session
through the store, which is how the file-overlap ground truth is compared.

### Results

300 sessions added, 89 scored queries. `P1` is the deterministic baseline;
later stages are additive over it.

Measured on the development GPU (RTX 5060 Laptop, 8 GB) with `device=auto`.
**These are session-unit numbers** (the ablation was run before the entry view
existed, and they were never re-derived in it); the entry-unit figures for P1 and
P2c are further down, and the gap is ~15 points at k=10:

| configuration | MRR | nDCG@10 | nDCG@100 | recall@10 | precision@10 | Add | Search |
|---|---|---|---|---|---|---|---|
| random baseline | 0.2334 | 0.1936 | 0.3804 | 0.2919 | 0.0972 | — | — |
| **P1** lexical + entity | 0.7248 | 0.6138 | 0.7047 | 0.6655 | 0.1972 | 56 s | 47 ms |
| P2a + dense | 0.7659 | 0.6421 | 0.7252 | 0.6885 | 0.2073 | 163 s | 224 ms |
| P2b + rerank | 0.7716 | 0.6474 | 0.7296 | 0.6914 | 0.2135 | 41 s | 521 ms |
| **P2c + dense + rerank** (default) | **0.7718** | **0.6586** | **0.7306** | **0.7133** | **0.2242** | 155 s | 674 ms |

MRR 0.77 against 0.23 for random, nDCG@10 0.66 against 0.19. Since the platform
consumes a token-counted *prefix* of our ranking, the `@10` window is the relevant
one — but read it as **10 sessions ≈ 21 memory entries**, not 10 entries. The
entry-unit reading of the same ranking is what the graded payload delivers, and it
is reported alongside from here on; see the section on the two accounting units.

### What recall@10 = 0.67 is actually made of

`recall@10` is the weakest-looking number in the table, so it is worth
decomposing before anyone tries to optimise it. `--dump-per-query` now exports
both the ranked session list and, for each relevant pair, *how many* files the
query's patch shares with that session. That splits the metric by label
strength (measured on the P1 run, `eval/results/current.json`; the label
structure is fixed, so the decomposition carries to other configurations):

| relevance | pairs | recall@10 | recall@100 |
|---|---|---|---|
| all (≥1 shared file) | 333 | 0.6655 | 0.9152 |
| **strong** (≥2 shared files) | 101 | **0.7750** | 0.9804 |
| ≥3 shared files | 10 | 0.9000 | 1.0000 |

The gradient is monotone and steep: the better the label, the better we rank it.
So the headline is not a ranking failure, it is mostly a **labelling** one:

- **70 % of "relevant" pairs share exactly one file** (232 of 333; mean overlap
  1.34, max 4). Relevance here is "touched at least one file the patch touched",
  which credits a session whose entire connection to the task is one file.
- **That one file is usually a repository hotspot**, not a coincidence of
  tests. The most-shared single files are `django/db/models/sql/query.py` (23
  pairs), `django/db/models/query_utils.py` (19), `sql/compiler.py` (13). Only
  15 % of weak pairs share a test file. So the definition inflates the relevant
  set wherever a repo has a file everybody touches: django averages **5.86**
  relevant sessions per query but only **1.67** strong ones.
- **43 of 99 queries have no strong-relevant session at all.**

It is also *not* a capacity or truncation artefact. The ceiling under perfect
ranking is 0.9814 (only 4 queries have more than 10 relevant sessions, 890
slots for 333 relevant pairs), and `candidate_pool` is 300 — the whole corpus —
so nothing is being cut before ranking.

What is left is the real target: of the 96 strong-relevant sessions, **33 % land
outside the top 10** (21.9 % at rank 11–20, 11.5 % beyond 20). That third is the
part worth attacking; the rest of the gap is measurement definition.

Practical consequence: treat `recall@10` on all pairs as a regression tripwire,
not a target, and read the strong-only figure alongside it. Optimising the
all-pairs number would mean learning to rank sessions whose only tie to the
query is a hot file — which is exactly the behaviour IDF exists to suppress.

**The strong-relevant gap is largely already closed by P2c.** The 33 % figure
above is measured on the P1 run. On the full default configuration (dense +
rerank, `eval/results/p2c_base.json`, GPU):

| configuration | strong recall@10 | strong outside top-10 | all recall@10 | MRR |
|---|---|---|---|---|
| P1 lexical + entity | 0.7750 | 33.3 % | 0.6655 | 0.7248 |
| **P2c + dense + rerank** | **0.8601** | **17.0 %** | 0.7115 | 0.7778 |

So the semantic channels already recover about half of what the lexical stack
misses. Of the 23 strong sessions still outside the top 10 under P2c, 7 are
never returned at all — those, not the 11–20 band, are what is left to attack.

**One attempt to close it from the lexical side: measured, failed, reverted.**
Session-pooled identifier evidence (`entity_session_share`): a chunk covers one
span of one file, so a session whose evidence spans two files arrives as two
middling chunks, and per-chunk scoring cannot tell it from a session that
matched once on a hot file. Pooling a session's identifier scores and crediting
each chunk `own + share × (session − own)` should restore that distinction.

It did not. On P1: MRR 0.7248 → 0.7340, strong recall@10 **0.7750 → 0.7750**,
strong-outside-top-10 33.3 % → 33.3 %. Paired bootstrap over 89 queries gives
**p ≈ 0.38** (14 queries moved, 9 up / 5 down) — indistinguishable from noise.
Reverted rather than shipped.

The reason is diagnostic, and it generalises. For the dropped strong sessions:

- **46 % contain none of the query's extracted identifiers at all** (17 of 37),
  so no amount of identifier weighting can reach them — only semantics can;
- of the 20 that *do* contain query identifiers, comparing their IDF-weighted
  identifier score against the highest-ranked irrelevant session in the same
  top-10 gives **5 wins, 5 losses, 10 exact ties**. Half the time the identifier
  evidence is literally identical, because every django session discusses the
  same ORM files.

That is the ceiling of the lexical/entity feature space on this corpus: the
label distinction ("shared two files" vs "shared one") is not a distinction the
text carries. It is also a caution for future work — the all-pairs `recall@10`
moved (+0.005) while the thing we set out to fix did not move at all, so any
change here has to be read on the strong-only split.

### Every quota in the config is counted in memory entries, not sessions

Adding the 300 benchmark sessions produces **28 675 memories** — a mean of
**95.6 entries per session** (median 89, max 261). That single number reframes
every retrieval quota, because all of them count entries:

| setting | value | in sessions |
|---|---|---|
| `recall_per_channel` | 120 | ~1.3 |
| `candidate_pool` | 300 | ~3.1 |
| `rerank_top_n` | 120 | ~1.3 |
| `listwise_max_candidates` | 40 | ~0.4 |

They were tuned while memories and sessions were roughly one-to-one, and they
were never re-derived after chunking. A single average session now holds enough
entries to fill a whole recall channel by itself.

**The entries are messages, not sub-message chunks.** Add chunks each message
independently (`target_chunk_tokens=320`, `hard_chunk_chars=24 000`), and on this
corpus that almost never splits anything: a message is a median of **151**
characters (mean 756), and **96.3 % of messages yield exactly one chunk**, only
3.7 % more than one. The count comes out *below* one entry per message (0.84)
because roughly a fifth of the messages flatten to empty text — `build_benchmark`
drops `thinking` blocks — and are skipped. So "one session ≈ 114 turns ≈ 95.6
memory entries" and "one turn ≈ one entry" are both true here, and the chunker is
effectively a pass-through. That would not hold for a corpus of large tool
outputs or file dumps, which is the case the chunker exists for.

**The 7 strong sessions never returned under P2c** split cleanly. Replaying the
search internals per pair: **5 of the 7 contribute zero entries to the candidate
pool** — they are lost at recall, not at the gate. The other 2 do reach scoring
and clear `min_evidence_score` comfortably (final 0.44 and 0.97), so they are
lost later, in P2c's cross-encoder stage, not to the noise gate.

**Widening the quotas does not fix it — measured.** `recall_per_channel=400`,
`candidate_pool=800` (≈8 sessions' worth of pool instead of ≈3):

| | pool 300 | pool 800 | delta |
|---|---|---|---|
| MRR | 0.7248 | 0.7186 | −0.0062 |
| recall@10 | 0.6655 | 0.6634 | −0.0021 |
| recall@100 | 0.9152 | 0.9143 | −0.0009 |

`recall@100` does not move, which is the tell: the missing sessions are not
sitting just outside the pool, they score too low to reach rank 100 even when
admitted. More candidates means more low-scoring entries competing in the same
normalisation, which is why the numbers drift slightly down rather than up. So
the cause is weak textual evidence, not truncation — consistent with the 46 %
of dropped strong sessions that contain none of the query's identifiers at all.

This does not mean the quotas are right; it means widening them is not the fix.
A quota expressed in *distinct sessions* (cap chunks per session before
truncating the pool) is the untested variant, and the pool figure above is the
measurement that would show whether it matters.

### The platform counts memory entries, not sessions

`api_guide.md` is explicit: `top_k` is "允许返回的最大记忆条数", fixed at 100 for
the scored run, and `data[]` entries are what "按返回顺序进入 Answer". The
platform has no concept of a session at the interface — it sees 100 text spans.

Neither do we, on the way out. `EvidenceItem` carries `memory_id`, `content`,
`score`, `created_at` — no session field — and `content` is a verbatim span with
no header or label. So the session never reaches the answer model: it only
decides **which** entries occupy the 100 slots and **in what order**. That is its
one real effect, and it is a packing policy, not information the answer model
reads.

Sessions are *our* unit, and the exchange rate is ours to set: Add turns one
session into ~95.6 memory entries, and `assemble` then caps a session at 3 of
them, so the observed packing is **2.06 memory entries per session**. That means
`top_k=100` buys roughly **48 sessions, not 100** — and with no per-session cap
it would buy one or two, since a single session alone holds enough entries to
fill the whole payload.

The two units do not score the same, and `run_benchmark` now reports both from a
single run — the unprefixed rows collapse `data[]` onto sessions, the `item_`
rows score the entry list itself. Default configuration (dense + rerank, GPU,
n=89):

| metric | session units | **memory-entry units** |
|---|---|---|
| recall@10 | 0.7344 | **0.5823** |
| nDCG@10 | 0.6787 | 0.5479 |
| precision@10 | 0.2334 | **0.4184** |
| MRR | 0.7922 | 0.7424 |
| recall@100 | 0.9118 | 0.9118 |
| items per session | — | 2.07 |

**The `item_` rows are the ones that track the platform.** The k=10 gap is not
cosmetic: 0.7344 versus 0.5823 means the session figure credits evidence sitting
in slots 11–25 that the graded payload does not deliver at a ten-slot budget.

Two rows read oddly and both are correct. **Entry precision is higher than
session precision** (0.4184 vs 0.2334) — a session holding three of the top ten
slots counts once in a session denominator and three times here, and that is the
metric doing its job: 41.8 % of the graded budget is spent on evidence from a
relevant session. And **the units agree exactly at k=100**, because that is the
whole payload: no truncation, no gap.

**Every number in this README above is in session units, which is the optimistic
of the two** — by ~15 points at top-10 on the default configuration. The session
figure asks "did any chunk of the right session make the window"; the entry
figure asks "how much of the right session's evidence is in the window", and that
is closer to what the answer model can actually read. They converge only at
k=100, where the whole payload is in view.

**But the gap is the window, not the signal.** Compare the two units at matched
payload size — session `recall@k` against entry `recall@k` where `k` is however
many entries those first `k` sessions actually consume:

| session k | entries those sessions read | session recall | entry recall, same entries |
|---|---|---|---|
| 5 | 14 | 0.5698 | 0.5730 |
| 10 | 25 | 0.6634 | 0.6794 |
| 20 | 42 | 0.7968 | 0.7621 |

Aligned, they agree to within ~4 points. So session `recall@10` is not measuring
something different from entry `recall@10`; it is entry `recall@25`. Both are
honest readings of the same ranking — at different budgets.

That is exactly why the unit has to be fixed, and why the session unit is the
wrong one for a headline number: **the size of a session-`k` window depends on
`max_evidence_per_session`.** At cap 1 those first ten sessions are ten entries;
at cap 3 they are twenty-five. So `recall@10` in a cap-1 run and `recall@10` in a
cap-3 run are not comparable quantities, which is the whole of the "cap 1 scores
marginally better" finding (0.9231 vs 0.9193) — a bigger window, not a better
ranking. An entry-`k` window is a fixed number of slots regardless of packaging,
so it survives a change of assembly policy and can be mapped onto the platform's
token prefix directly.

**What the metrics count, and what they deliberately do not.**

- `recall@k` takes a **set intersection** and `nDCG@k` credits a document only on
  **first appearance**, so padding the entry list with repeats of one session
  cannot inflate either. On the collapsed list this changes nothing, which is why
  the session numbers here are unchanged from before.
- `precision@k` deliberately does **not** deduplicate: it counts slots, because a
  repeated session consuming three of them is a real cost to the answer model's
  budget. Set-deduplicating it would silently redefine it as "density of distinct
  relevant sessions" and report a payload wasting two thirds of its budget on
  repeats as though the slots were free — measured, that is 0.4184 versus 0.1511
  on the default configuration.
- All of them are still **binary**: relevance is `1.0 if doc_id in relevant else
  0.0`. The `score` field we return is never read, and there is no graded
  relevance, so nothing is scored for *quality* of the returned text — one
  perfect span and three identical spans score the same.

One consequence worth spelling out even now that both units are reported: **the
size of a session-`k` window still depends on `max_evidence_per_session`.** At
cap 1 those first ten sessions are ten entries; at cap 3 they are twenty-odd. So
session `recall@10` in a cap-1 run and in a cap-3 run are not comparable
quantities — which is the whole of the "cap 1 scores marginally better" line in
the tuning table (0.9231 vs 0.9193): a bigger window, not a better ranking. An
entry-`k` window is a fixed number of slots regardless of packaging, so it is the
one to put in a headline.

This is also the same gap the evidence metric found from the other direction: the
answer session is present in the payload 100 % of the time, but `decidable` is
0.233. Session-level presence is not entry-level usefulness.

### Where the entry-level loss comes from

`scripts/diagnose_entry_recall.py` splits the relevant sessions of every query
into the window, the payload below it, or nowhere — under the real slot
constraint, since a session spends up to `cap` of the k slots and the window
therefore cannot hold `min(avail, k)` distinct sessions. On the default
configuration, with entry `recall@10 = 0.5823` (n=89, 333 relevant sessions):

| bucket | macro | micro |
|---|---|---|
| in window (achieved) | 58.2 % | 39.6 % |
| in payload, recoverable by reordering | 24.8 % | 25.5 % |
| in payload, no slot left at cap 3 | 8.2 % | 18.3 % |
| never entered the payload | 8.8 % | 16.5 % |
| ceiling (perfect reorder, same payload) | 83.0 % | 65.2 % |

Of the 42-point shortfall, **reordering owns 59 %, retrieval 21 %, and the
per-session cap 20 %** — in the macro view, which is the one that matches how the
platform scores, since each query is one answer. Weighted by relevant sessions
instead the three are near even (25.5 / 18.3 / 16.5), and that is django/django,
which carries 211 of the 333 relevant sessions. The micro column is not a
statement that those queries matter more.

Two facts fix the shape of this. **Candidate generation is not the bottleneck**:
90.7 % of relevant sessions are already in the payload. And **the payload is
thinner than `top_k` suggests** — 59.2 entries over 28.6 sessions, because the
0.15 noise gate ends assembly long before the 100-item budget is reached
(`evidence.py` breaks on `group[0].final < min_evidence_score`, not on
`len(items) >= top_k`).

**The cap is a mechanical lever, not a ranking effect.** The session set is
gate-decided, so lowering `max_evidence_per_session` costs no retrieval — it only
changes how many sessions the first k slots can span:

| cap | item recall@10 | sessions spanned by 10 slots | equals |
|---|---|---|---|
| 3 (default) | 0.5823 | 4.00 | session recall@4 = 0.5818 |
| 2 | 0.6321 | 5.19 | session recall@5 = 0.6313 |
| 1 | 0.7344 | 9.42 | session recall@9 = 0.7253 |

The cap-3 row reads the dump verbatim; the other rows are replayed by rebuilding
the entry sequence from `ranked`, which `--dump-per-query` retains for exactly
this purpose, and the replay reproduces the measured 0.5823 to the digit. This is
the whole of the 15-point gap between the two units: entry `recall@10` is not a
stricter reading of the same ranking, it is session `recall@4`.

`--cap N` was added to `run_benchmark.py` so that this is a flag rather than an
edit. Note that the harness constructs `Settings` from dataclass defaults plus its
own overrides and never calls `from_env`, so **editing `.env` has no effect on
`run_benchmark.py`** — the flag is the way in. Before moving the default, check
the end-to-end answer score: the session view barely moves across caps, and cap 1
hands the answer model a single span per session, which no entry-view metric can
price.

**"Too much noise" is largely a misreading.** 58.2 % of the window is irrelevant,
but only part of that is crowding. Reordering the same payload lifts the share of
slots holding a relevant entry from 41.8 % to 63.3 %; the remaining 36.7 % is not
displaced evidence but absent evidence — the mean query has 3.74 relevant sessions
and 8.06 relevant chunks in the payload against 10 slots. The window is larger
than the answer.

**The value of dense retrieval depends on the device, and this flipped the
decision.** On CPU it added no metric gain over rerank-only while costing ~5× the
Add time. On GPU it does help — nDCG@10 0.6586 vs 0.6474, recall@10 0.7133 vs
0.6914 — and the Add cost is 155 s for 300 requests (~0.5 s each), far inside the
30-minute contract budget. Since `device` defaults to `auto`, a GPU host gets the
better ranking and a CPU-only host still fits comfortably. Dense is therefore on
by default, with that hardware dependency stated rather than hidden.

Device sensitivity, measured on the same corpus:

| stage | CPU | GPU | speedup |
|---|---|---|---|
| embedding (256 long documents) | 14 docs/s | 112 docs/s | 8× |
| reranking (120-document pool) | 15.9 ms/doc | 2.0 ms/doc | 8× |

Caveat that cuts the other way: the proxy's file-overlap ground truth
**structurally cannot credit** a session that helps semantically without sharing
files — which is exactly dense retrieval's strength. So the dense figures here
are a lower bound, and its advantage is probably understated rather than
overstated.

### Payload shape did not affect ranking

An earlier revision padded ``content`` with a ``[memory]/[file_path]/[time]``
header and filled ``created_at`` with our own write time. Both were wrong against
the contract (see `docs/DESIGN.md`), and both were removed: content is now a
verbatim span of stored memory text and ``created_at`` carries the source
timestamp.

Re-measured afterwards, the ranking is unchanged — P1 MRR 0.7248 and recall@10
0.6655, P2c MRR 0.7718 and recall@10 0.7133, matching the pre-change figures. The
reason is that a real session trajectory is already close to maximally compact,
so the header was costing answer-model budget without buying ranking quality.

### Session-major pool and session-level reranking: measured, not shipped

The two variants above were built on one observation: every quota counts
entries, and a session is ~95.6 entries, so `candidate_pool` 300 spans ~3
sessions and `rerank_top_n` 120 barely 1.3. The cross-encoder was therefore
spending its whole budget re-ordering chunks *inside* one or two sessions.

Two switches were added, both off by default:

* `candidate_per_session` + `recall_channel_depth` — pull a deeper slice per
  channel, then keep at most N entries per session before truncating the pool
  (`Retriever._cap_per_session`);
* `rerank_session_level` — score one representative document per session and
  apply that score to every candidate of the session, so only the session order
  moves (`SearchPipeline._rerank_sessions`).

All rows are the same machine, same corpus, n=89, dense + rerank on GPU:

| configuration | MRR | nDCG@10 | recall@10 | recall@100 | item MRR | item nDCG@10 | item recall@10 | items/session |
|---|---|---|---|---|---|---|---|---|
| default (cap 3) | 0.7922 | 0.6787 | 0.7344 | 0.9118 | 0.7424 | 0.5479 | 0.5823 | 2.07 |
| A: session-major pool | 0.8109 | 0.6827 | 0.7319 | 0.8907 | 0.7682 | 0.5520 | 0.5628 | 2.48 |
| A+B: + session rerank | 0.7334 | 0.6189 | 0.6816 | 0.8724 | 0.6778 | 0.4939 | 0.5368 | 2.46 |
| A+B, rerank weight 0.30 | 0.7509 | 0.6422 | 0.7006 | 0.9160 | 0.6915 | 0.5214 | 0.5731 | 2.53 |
| cap 2 (default pool) | 0.7922 | 0.6787 | 0.7344 | 0.9142 | 0.7579 | 0.5869 | **0.6321** | 1.62 |
| A + cap 2 | 0.8111 | 0.6827 | 0.7319 | 0.9381 | 0.7817 | 0.5882 | 0.6122 | 1.83 |

Paired bootstrap over 89 queries (search latency is not compared: it ranged
1.06–1.49 s across runs on this host, so the differences are host noise):

| comparison | MRR delta | item recall@10 delta |
|---|---|---|
| A − default | +0.0187, p=0.24 | −0.0194, p=0.25 |
| A+B − default | **−0.0588, p=0.006** | **−0.0454, p=0.015** |
| A+cap2 − cap2 | +0.0190, p=0.25 | −0.0199, p=0.30 |
| cap 2 − cap 3 | +0.0000 (0 queries moved) | **+0.0498, CI [+0.024, +0.083], p<0.001** |

**B is a measured loss, and it fails the attenuation test.** Dropping the
blend weight from 0.65 to 0.30 walks every metric monotonically back toward the
baseline (MRR 0.7334 → 0.7509, item recall@10 0.5368 → 0.5731) — the same
signature listwise showed, and the same reading: a stage contributing noise
rather than signal. The likely mechanism is the representative document: one
chunk does not stand for a 96-entry session, and the listwise experiment above
already showed that a judge needs more excerpt, not less, to decide relevance.
So the cross-encoder's value here is picking the best chunk *within* a session,
not ordering sessions against each other.

**A is not provably a gain.** MRR moves +0.019 with p=0.25 while item
recall@10 moves −0.019 with p=0.25 — the same size in both directions, which is
noise. It does widen the payload (recall@100 0.9118 → 0.9381 under cap 2), but
that is the one thing the earlier pool-300→800 experiment already showed is not
the binding constraint, and it costs entry-window coverage because a session
that now has three candidates in the pool fills more of the ten slots
(items/session 1.62 → 1.83). Candidate generation is not the bottleneck, again.

**The one significant result in this block is the cap, and it is mechanical.**
`max_evidence_per_session` 3 → 2 lifts item recall@10 by +0.0498 (p<0.001) with
zero queries changing their session-level MRR — it cannot be a ranking effect,
it only changes how many sessions the first ten slots span.

Whether that is worth having is a different question, so it was priced with
`run_evidence.py` (n=30, same machine, `decidable` = the decisive line is in
what the answer model can read and no distractor contradicts it):

| cap | items | tokens | decidable @1000 | @2000 | @4000 | @8000 | all | decisive (all) |
|---|---|---|---|---|---|---|---|---|
| 1 | 34.0 | 3981 | 0.400 | 0.400 | 0.367 | 0.333 | 0.333 | 0.500 |
| 2 | 54.1 | 5557 | 0.433 | 0.433 | 0.400 | 0.300 | 0.300 | 0.633 |
| **3 (default)** | 67.2 | 6548 | 0.433 | 0.433 | **0.433** | **0.367** | **0.333** | **0.700** |
| 4 | 75.1 | 6996 | 0.433 | 0.400 | 0.400 | 0.367 | 0.400 | 0.733 |

cap 1 is a clear loss — one span per session drops decisive evidence from 0.700
to 0.500, which is the failure README already suspected. Between 2, 3 and 4 the
differences are one or two questions out of thirty, i.e. inside the noise, but
every prefix row of cap 3 is ≥ cap 2, and `decisive` falls monotonically as the
cap tightens (0.733 → 0.700 → 0.633 → 0.500).

So cap 2 buys +0.0498 on a *proxy of window coverage* and pays for it on the
proxy of *usefulness*, and the payment is on the axis the platform actually
scores. It is not adopted; the default stays at 3.

(Compare within this table only: the `decidable` 0.233 baseline quoted further
up was measured on an earlier revision, and the same command now returns 0.333
at cap 3. The four rows here are one machine, one revision, one command.)

### Listwise reranking (LLM): measured, not shipped

A final stage where gpt-4o-mini reads the whole candidate list and scores it
comparatively. A cross-encoder scores each pair in isolation; a listwise read can
see that forty candidates all come from the same repository and only one explains
the failure.

It is implemented, tested, and **off by default**, because it did not earn its
cost here:

| configuration | MRR | nDCG@10 | recall@10 | precision@10 | Search |
|---|---|---|---|---|---|
| P2c, no listwise | **0.7800** | 0.6687 | **0.7186** | 0.2242 | 0.8 s |
| + listwise, excerpt 700, w 0.5 | 0.7683 | 0.6357 | 0.6746 | 0.2140 | 4.5 s |
| + listwise, excerpt 1800, w 0.5 | 0.7686 | 0.6399 | 0.6919 | 0.2174 | 5.3 s |
| + listwise, excerpt 1800, w 0.25 | 0.7656 | 0.6594 | 0.7061 | 0.2230 | 5.6 s |

Two things to read from this. **Attenuating the weight drifts the numbers
monotonically back toward the baseline** — the signature of a stage contributing
noise rather than signal. If it carried signal, some weight would beat the
baseline; none does. And **the excerpt size matters** (700 → 1800 improved
recall@10 from 0.675 to 0.692), confirming that clipping a whole chunk discards
the text that decides relevance.

**The confound, stated plainly.** Our ground truth is "touched the same file",
while the judge optimises "would help answer the question" — different objectives.
Inspecting a case by hand showed the judge scoring `git clone` boilerplate 0/10
(*correct* — it answers nothing) while our recall had supplied no relevant memory
at all. The proxy therefore penalises the judge for disagreeing with it, and the
drop partly reflects *fewer returned items* after the noise gate removed what the
judge correctly rejected, not worse ordering.

So the honest position is "unproven on this benchmark", not "useless", and the
stage stays switchable. Settling it needs the end-to-end Answer/Eval harness
(platform Answer model + judge), which does not exist yet; that is the next piece
of work, and listwise should be re-measured there before any decision to enable
it.

### End-to-end Answer evaluation (case study, not a tuning signal)

> Superseded for tuning by the deterministic metric above. Kept because it is
> how the content-selection problem was found, and because the same harness is the
> right tool once the platform's own Answer and Eval run.

Retrieval metrics show whether a session was found; the platform scores whether
the returned memory let the answer model *solve the task*. `eval/run_endtoend.py`
measures the second thing.

**Design.** `eval/build_qa.py` builds 90 multiple-choice questions whose answers
are objective — given the issue text, which file must be changed? The gold answer
comes from the task's own patch, and distractors are files touched by *other*
tasks in the same repository, so they are plausible rather than obviously wrong.
Gold positions are balanced (17/23/23/27 across A–D), so position carries no
signal. No judge is involved in scoring, which avoids the self-preference bias of
one model answering and grading.

Each question is answered twice: with no memory (the model's own prior) and with
our retrieved memories supplied in returned rank order as a token-counted prefix,
mirroring the platform. **Only the difference attributes anything to the memory
system**, because absolute accuracy is dominated by what the model already knew.

**Result: no detectable memory contribution.**

| condition | n | accuracy | mean items shown |
|---|---|---|---|
| no memory | 90 | 0.689 | — |
| memory, top_k=100 (wide context) | 90 | 0.689 | 67.4 |
| memory, top_k=10 (tight context) | 90 | 0.700 | 10.0 |

The wide condition moved 10 answers — **4 gained, 4 lost** — a net zero that
McNemar's test cannot distinguish from chance (b=4, c=4, χ²=0.125, p=0.724). Tight
context moved the result by +0.011, also negligible.

The memory *was* reaching the model: every question received a non-empty context
(mean 67 candidates at top_k=100), and 80 of 90 included a session that shares a
file with the task's patch. So this is not a plumbing failure — it is a null
result for this question type.

**How the answer model locates the file without memory.** Measured, not assumed,
because the initial explanation ("it knows the repository") turned out to be only
half right:

| mechanism | evidence |
|---|---|
| Vocabulary mapping: issue words → path words | A trivial lexical baseline (pick the option with the greatest token overlap with the issue) scores **0.500** against 0.250 random. |
| Prior knowledge of well-known libraries | Asked why, the model answers in those terms: *"this file is responsible for model validation checks in Django, including detecting duplicate `db_table` settings"* (→ `django/core/checks/model_checks.py`), and *"the question discusses `BaseFormSet` and its `empty_form` ... located in `django/forms/formsets.py`"*. Both are library knowledge, not repository history. |

The per-repository spread tracks how deeply each library is represented in
pretraining: django 0.806, sympy 0.769, matplotlib 0.750 versus scikit-learn
0.300. Literal mention of the answer in the issue explains a further slice
(stem mentioned in 27.8% of questions: accuracy 0.840 there, 0.631 elsewhere).

**Neither mechanism is something a memory system supplies.** That is a flaw in
this question type: a file-location answer is largely derivable from the issue
text, so there is little for retrieved experience to add.

**The aggregate hides two opposing effects.** Splitting by whether the model
would have got it right unaided is far more informative than the raw mean:

| subset | n | no memory | with memory | delta |
|---|---|---|---|---|
| model already correct | 62 | 1.000 | 0.935 | **−0.065** |
| model wrong without memory | 28 | 0.000 | 0.143 | **+0.143** |

Memory *helps exactly where it should* — on questions the model cannot answer
alone — and *hurts on questions it already handles*, by displacing correct
reasoning with same-repository noise. Those effects cancel to 0.000 overall.
With 4 questions in each direction, neither is statistically significant, so this
is a directional finding, not a result.

The actionable implication is a design constraint rather than a number: returned
evidence must be gated hard enough that it cannot displace correct prior
reasoning, while still surfacing when the model is otherwise stuck. That is
precisely the "relevant versus noisy" tension the real track tests, and it is
where the next iteration should focus.

**What would actually settle it.** File localisation is too easy in the wrong way.
A diagnostic question type needs an answer that is *not* derivable from the issue
text — for example, asking how a specific past session handled a situation, where
the answer is that session's approach. The definitive measure is producing real
patches and running the SWE-bench test harness (`FAIL_TO_PASS` / `PASS_TO_PASS`),
which needs repository checkouts and per-task Docker images; the infrastructure is
in `benchmark/SWEContextBench/swebench_memory/harness/`.

### Why the answer model gets 0.333: the decisive line is dropped

The session-artifact questions have a clean validity check (no-memory 0.233 ≈
chance 0.25), so the +0.100 from memory is real signal. But the retrieved answer
session is in context **100% of the time** while accuracy is only 0.333, which
looks like the model ignoring available evidence. It is not.

The decisive evidence is a tool-call record, stored verbatim:

```
[tool Edit] {"file_path": ".../sklearn/mixture/base.py", "old_string": "..."}
[tool Read] {"file_path": ".../sklearn/mixture/base.py", "limit": 30}
```

`Edit` versus `Read` is exactly what the question asks, and it is present in the
text. `scripts/diagnose_edit_signal.py` measures how often it survives into what we
return, counting all three forms the evidence can take (the tool call, the
`has been updated` result line, and the `diff --git` header):

| measurement | value |
|---|---|
| answer session retrieved | 30/30 (100%) |
| gold file appears anywhere in returned evidence | 20/20 (100%) |
| gold file shown as *modified* in returned evidence | **6/20 (30%)** |
| a distractor shown as modified (would make it ambiguous) | 0/20 (0%) |

30% against a measured accuracy of 33.3% is the whole story: **the model answers
correctly exactly when the line that settles the question survives our
truncation.** The bottleneck is evidence selection, not recall and not the answer
model. We retrieve the right session and then cut away the part that is useful.

This is actionable. Our per-item window (~380 tokens) is chosen for token budget,
not for what carries the answer, and in a long trajectory the decisive fragment is
one line among thousands. Selecting windows by query-term density was a first step;
what this measurement shows is that retrieval units need to be chosen so that
*operative* content — the change actually made — is preferentially retained.

An earlier version of this diagnostic searched for the file name as a substring and
reported 45%, which was a false positive rate: an Edit payload's `old_string` can
name other files. Parsing the `file_path` field and counting the three evidence
forms is what made the number trustworthy.

**The item token budget was the knob, and 380 was the bottleneck.** Same
harness, same 30 questions, sweeping `evidence_item_tokens` (the cap on an item
rendered in full form):

| item tokens | payload tokens | decidable @1000 | @2000 | @4000 | @8000 | all | decisive (all) | ambiguous (all) |
|---|---|---|---|---|---|---|---|---|
| 380 (old default) | 6548 | 0.433 | 0.433 | 0.433 | 0.367 | 0.333 | 0.700 | 0.433 |
| 600 | 6942 | 0.533 | 0.467 | 0.500 | 0.433 | 0.367 | 0.800 | 0.467 |
| **800 (new default)** | 7190 | **0.600** | **0.567** | **0.567** | **0.467** | **0.400** | 0.800 | 0.433 |
| 1200 | 7530 | 0.500 | 0.533 | 0.567 | 0.433 | 0.367 | 0.800 | 0.467 |

Paired over the same questions, 800 versus 380: **+3 decisive / −0** (exact
p=0.25), **+2 decidable / −0** (p=0.50), and zero questions moved on
`ambiguous`. It is a one-directional change — nothing gets worse, three
questions get better — and it costs +9.8 % payload tokens (6548 → 7190, still
an eighth of `evidence_budget_tokens`). 1200 buys nothing over 800 and starts
pulling distractor modifications into the window.

Two neighbouring knobs were swept and do **not** earn a change:

* `evidence_full_count` 8 → 40 (more items in full form): tokens 6548 → 10376
  for `decidable` 0.433 → 0.400 at @4000 and 0.400 at the whole payload. Wide
  but shallow loses to narrow but deep.
* `evidence_operative_weight` 1.0 → 3.0: `decisive` falls 0.700 → 0.667 while
  `ambiguous` rises 0.433 → 0.533 — weighting operative lines harder pulls in
  *other* sessions' modifications too, which is exactly the ambiguity the
  metric exists to catch.

**The ranking is untouched by any of this**, which is what makes it a clean
content-selection win: re-running the proxy benchmark at 800 reproduces every
number to the digit (MRR 0.7922, recall@10 0.7344, item recall@10 0.5823,
items/session 2.07, 59.2 items returned). Only the text each item carries
changes.

### Evidence sufficiency: the deterministic metric to tune against

`eval/run_evidence.py` measures what our Search output actually carries, with no
model in the loop. For each question it parses the returned content and asks:

| metric | meaning |
|---|---|
| answer session retrieved | did recall find the right session at all |
| decisive evidence present | does the returned content show the **operative** action for the gold artifact (a tool call, an update notice, or a diff header) |
| ambiguous | does such a marker also appear for a distractor, so the evidence cannot settle the question |
| **decidable** | decisive present **and** not ambiguous — the number to raise |

Everything is string parsing, so it is reproducible, fast and free. Baseline:

| metric | value |
|---|---|
| answer session retrieved | **1.000** |
| decisive evidence present | 0.500 |
| ambiguous | 0.467 |
| **decidable** | **0.233** |

(The 0.300 / 0.367 figures recorded earlier were measured before assembly became
session-major; re-measured on the shipped code they are 0.233 / 0.467. Read the
next section before trusting any of them — this table concatenates the whole
payload, which is not what gets graded.)

Two things follow. **Recall is not the problem** — the right session is in context
every time, which is why adding more retrieval channels would be wasted effort.
The losses are entirely in what we return: half the time the decisive line is cut
away, and in nearly half of cases evidence from *another* session makes a
distractor look modified too.

`decidable` = 0.233 is in the same range as the non-deterministic answer accuracy
measured earlier (0.333), which is the validation that it measures the same thing
— while being reproducible.

**Operative weighting, swept on the deterministic metric:**

| `evidence_operative_weight` | decisive | ambiguous | decidable |
|---|---|---|---|
| 0.0 (query terms only) | 0.467 | 0.333 | 0.233 |
| **1.0 (default)** | 0.500 | 0.367 | **0.300** |
| 3.0 | 0.500 | 0.467 | 0.200 |

The mechanism is coherent: weighting operative lines pulls in **all** edit markers,
including ones belonging to *other* sessions that touch a distractor, so beyond
1.0 the ambiguity it introduces outweighs the decisive evidence it recovers. The
default of 1.0 is the best of the three tested.

**Ambiguity is now the dominant error mode (0.367).** That is the next thing to
attack: our returned items mix evidence from many sessions with no way for the
reader to tell which session a marker belongs to, so another session's edit can
contradict the answer. n=30 means one question is 0.033, so these differences are
directions with a mechanism, not established effects.

### Assembly: sessions rank, chunks are evidence

`assemble` used to walk the globally sorted chunk list and count how many items
each session had already taken, which conflates two decisions: which session is
the answer, and which text represents it. It now groups candidates by session
first, so the session's head chunk *is* its score, the noise gate applies to it,
and within a session the operative chunk can be promoted into the first slot
(`evidence_operative_promotion`, `-1` = all sessions, `0` = never;
`evidence_max_sessions` caps distinct sessions, `0` = unlimited).

Promotion targets a measured loss: a session yields ~47 tool-call chunks that are
near-identical to each other — the same file read five times and edited once — and
they score alike because the paths carry the term weight while the verb `Edit`
carries none. Taken by score alone, the edit sits eighth in its own session and
never reaches a slot. It is also absent from every query vocabulary: the word
`edit` appeared in **0 of 30** questions and options.

Two predictions were tested and one failed:

| `max_sessions` | promotion | session retrieved | decisive | ambiguous | **decidable** | items |
|---|---|---|---|---|---|---|
| 0 (unlimited) | all | 1.000 | **0.667** | **0.567** | 0.267 | 67.1 |
| 0 | top 3 | 1.000 | 0.633 | 0.500 | 0.267 | 67.1 |
| **0** | **top 1 (default)** | 1.000 | 0.567 | **0.333** | **0.333** | 67.1 |
| 0 | never | 1.000 | 0.500 | 0.367 | 0.300 | 67.1 |
| 5 | top 1 | 0.933 | 0.400 | 0.167 | 0.333 | 13.9 |
| 3 | top 1 | 0.900 | 0.333 | 0.100 | 0.267 | 8.6 |
| 1 | any | 0.700 | 0.300 | 0.000 | 0.300 | 3.0 |

**Promotion works on its own axis and fails on the aggregate.** Promoting in every
session recovered decisive evidence 0.500 → 0.667 exactly as predicted, and raised
ambiguity 0.367 → 0.567 at the same time, because promoting an action surfaces
*other* sessions' actions too. Restricting promotion to the single session we
consider most likely to be the answer is the only setting that improved on all
three axes over the baseline simultaneously: decisive 0.567, ambiguous 0.333,
decidable 0.333.

**Capping sessions to buy precision did not work.** The plan was to spend
coverage — we had it at 1.000 — on removing contradiction. Measured, coverage is
not as free as it looked: the answer session is present somewhere in 67 items but
is the *top-ranked* session in only 70% of questions, so tightening the cap loses
decisive evidence as fast as ambiguity and `decidable` stays flat at 0.27–0.30 all
the way down. The session ranking, not the assembly policy, is the binding
constraint.

The cap is still worth knowing about for a different reason: at `max_sessions=1`
ambiguity is 0.000 and `decidable` equals the baseline's 0.300 while the payload is
**3 items instead of 67** — the same answer-carrying quality for a twentieth of the
tokens. That matters under a real input window, which our metric does not model:
`run_evidence.py` concatenates everything we return, so it scores tail items the
platform may never show the answer model. Making the metric honour a token prefix
is the next measurement to build, because it is the graded object.

**Cost on the retrieval proxy: none.** Re-measured with the new assembly at the
shipped defaults, against the file-overlap ground truth:

| metric | before | after |
|---|---|---|
| MRR | 0.7800 | 0.7800 |
| nDCG@10 | 0.6687 | 0.6686 |
| nDCG@100 | 0.7306 | 0.7368 |
| recall@10 | 0.7186 | 0.7186 |
| precision@10 | 0.2242 | 0.2242 |
| mean items returned | ~67 | 59.2 |

Ordering by session and promoting the operative chunk is free on ranking metrics
that cannot see the difference — file overlap does not care which of a session's
chunks is first — and it is not free on the metric that can.

### Scoring only what the answer model can read

The metric above concatenated **everything** we return. That is not the graded
object: the platform feeds the answer model a token-counted *prefix* of `data[]`
in our order, so evidence sitting in item 60 of a 66-item payload is not evidence
anything reads. Worse, the gradient was wrong — a policy that returns more items
could only score higher, never lower, which points a denoising system the wrong
way. `run_evidence.py` now scores a ladder of token budgets, taking items in
emission order and cutting the straddling one mid-way:

    python eval/run_evidence.py --prefix-tokens 1000 2000 4000 8000

At shipped defaults (30 questions, mean 65.7 items / 6 518 tokens returned):

| prefix | items visible | decisive | ambiguous | **decidable** |
|---|---|---|---|---|
| 1 000 tokens | 5.5 | 0.267 | 0.067 | **0.233** |
| 2 000 | 13.5 | 0.367 | 0.200 | **0.233** |
| 4 000 | 33.9 | 0.433 | 0.367 | **0.233** |
| 8 000 | 60.0 | 0.500 | 0.467 | **0.233** |
| all (upper bound) | 65.7 | 0.500 | 0.467 | 0.233 |

Two things fall out.

**The old headline number was inflated by the tail.** `decisive_present = 0.500`
was reported as "half the time the decisive line survives"; under a 1 000-token
prefix it is 0.267. The other 0.233 of credit came from items 6–66.

**`decidable` is flat at 0.233 across the whole ladder.** The tail is neither
neutral nor a trade-off — it adds decisive evidence and contradiction in lockstep,
so it moves questions from "neither" straight to "both" and never to "decidable".
The same answer-carrying quality is reached at 5.5 visible items as at 65.7. That
is the strongest argument yet for spending the budget on *session ranking* rather
than on breadth: most of the returned items are, on this measurement, doing
nothing but contradicting each other.

The `all` row is kept for comparison with earlier runs and is labelled an upper
bound, not a result.

**The compact payload matches it.** Re-run with `evidence_max_sessions=1`:

| payload | items | tokens | decisive | ambiguous | **decidable** |
|---|---|---|---|---|---|
| shipped defaults | 65.7 | 6 518 | 0.500 | 0.467 | **0.233** |
| `max_sessions=1` | 2.9 | 676 | 0.267 | 0.067 | **0.233** |

Three items score the same as sixty-six on the graded metric, at a tenth of the
tokens, and they fit inside a 1 000-token prefix so they are not prefix-sensitive
at all. Counting from the rates: the tail adds decisive evidence on 7 more
questions and adds a contradicting distractor on 12 more, and `decidable` does not
move — so every question where the tail contributes evidence is a question where
it also contributes noise. `session_retrieved` falls to 0.533, as it must when
only one session is shown, and `decidable` is unchanged, which says the questions
lost were never decidable.

This does not yet reverse the shipped default. `decidable` is a proxy: it cannot
see whether a stronger answer model resolves an ambiguity our string parser
cannot, and the platform's real prefix length is not published, so a large prefix
would show 0.500 decisive rather than 0.267. What it does establish is that
breadth is not currently buying anything measurable, and that ambiguity — the
dominant error mode — is 7x lower in the compact payload. The decision to leave
`evidence_max_sessions=0` should be re-tested against an end-to-end Answer run
before it is treated as settled.

### Chunk kind labels, audited against the corpus

The chunker labels every segment as one of eight structural kinds, and the label
is load-bearing: it selects the intent/kind scoring bonus and it decides whether
the aggressive symbol/command/package entity extractors run on that text
(`code/diff/stacktrace/test/config` are "codeish", prose is not). Nothing so far
checked whether those eight kinds actually occur, or occur *correctly*, on real
trajectories. `scripts/audit_chunk_kinds.py` does: it runs the shipped chunker
over `eval/data/benchmark.json` (300 sessions, 34 186 messages, 486 049 lines,
37 703 chunks) and reports the distribution plus two suspect-label counts.

Why the distribution is lopsided is visible in the same run: **83.2% of corpus
lines sit inside code fences and 61 035 of 61 523 fences (99.2%) carry no
language tag**, so the language-driven branches of the fenced classifier almost
never fire and unlabeled blocks fall through to `code`.

| kind | before | after | what changed |
|---|---|---|---|
| `code` | 30 224 (80.2%) | 30 101 (81.5%) | absorbs nothing new; still the bulk |
| `prose` | 5 046 | 4 970 | net of gaining mislabelled lists and prompts, losing packed boundaries |
| `diff` | 1 103 | 646 | 475 chunks had **no diff header at all** — markdown bullets start with `-`, so they counted as changed lines |
| `stacktrace` | 990 | 990 | unchanged |
| `config` | 262 | 10 | **249 were the task prompt** (`instance_id:`, `problem_statement: <sentence>`), which also mis-set their `lang` to `yaml` (250 → 5) and switched on codeish entity extraction over issue prose |
| `test` | 38 | 172 | pytest output inside unlabeled fences is now recognised by content (a run marker plus ≥50% test-shaped lines); test *source* still stays `code` |
| `cmd` | 38 | 38 | unchanged; 14 of them (37%) still open with a whitelisted command word rather than a shell prompt — left alone, 38 chunks cannot justify the risk |
| `log` | 2 | 2 | the corpus genuinely has almost no timestamped log burst — only 12 fenced blocks reach a 50% log-line ratio in the first place. This kind is rare here, not mislabelled |

All eight kinds do occur, so no rule is dead code — but four of them
(`test`, `cmd`, `config`, `log`) were either nearly empty or mostly wrong, and
`config` was the worst: 95% of its chunks were one mislabelled artefact repeated
across sessions.

**The metrics did not reward this.** Measured on the same machine with the same
deterministic configuration (dense and rerank unavailable, so the P1 path):

| | baseline | chunker fix | chunker + `code` bonus |
|---|---|---|---|
| MRR | 0.7800 | 0.7736 | 0.7778 |
| nDCG@10 | 0.6686 | 0.6611 | 0.6566 |
| recall@10 | 0.7186 | 0.7214 | 0.7115 |
| precision@10 | 0.2242 | 0.2323 | 0.2267 |
| `decidable`, unlimited row | 0.367 | 0.400 | 0.400 |
| `decidable`, mean of prefix rows | 0.417 | 0.408 | 0.383 |

The chunker change is flat-to-mixed: `recall@10` and `precision@10` improve,
`nDCG@10` and MRR slip by less than one query's worth, and on a 30-question set
every `decidable` difference is 1–2 questions. It was kept anyway, on the ground
that a mislabelled chunk is an index defect rather than a tuning knob — it changes
which entity extractors run, and the proxy's file-overlap truth cannot price
that. Two things to re-test if this is revisited: whether `prose` deserves a
debug bonus (the demoted lists and prompts *are* the sessions' own summaries of
what they changed), and whether the corpus is simply too small to resolve
label-quality effects at all.

The speculative part was the `code` entry in the debug intent table, motivated by
operative evidence living in `code` chunks (2 869 of the 3 464 chunks carrying an
operative line, versus 594 in `diff`). It cost the metrics that decide what the
answer model reads — recall@10 0.7214 → 0.7115, precision@10 0.2323 → 0.2267, and
the mean across the four prefix `decidable` rows 0.408 → 0.383 (three of four
budgets down, 8 000 tokens flat) — while only MRR improved, 0.7736 → 0.7778. The
reason is structural: `code` is 81% of the corpus, so a kind-level weight cannot
isolate a 9% minority inside it. **Dropped.** Whatever promotes operative evidence
has to read the chunk's lines, not its label — which is what
`evidence_operative_weight` already does for span selection and has not yet been
tried for ranking.

### Tuning decisions taken from measurements, not intuition

| Decision | Evidence |
|---|---|
| Cap items per session at 3 | Recall@100 rose 0.845 → 0.919 and nDCG@100 0.681 → 0.705. Without a cap, ~100 returned chunks collapsed to ~23 distinct sessions, starving other relevant work. |
| Reject the optimum at cap=1 | cap=1 measured marginally better recall (0.9231 vs 0.9193) but the proxy scores *whether a session was found*, not *whether its content is enough to answer*. Optimizing a measurable proxy at the cost of an unmeasurable quality is how benchmarks get gamed; cap=3 keeps session context for a 0.4 % metric difference. |
| Rerank weight 0.65, temperature 2.0 | Both swept. Weight: 0.65 peaks (MRR 0.7716); 0.85 and 0.95 degrade (0.7311, 0.7347) even though precision@10 rises — precision@10 is not what the answer model needs. |
| End-to-end result reported as null, not as a win | 0.689 vs 0.689 with 4 gained / 4 lost and p=0.72 is indistinguishable from chance. Reporting the aggregate as anything other than "no detectable effect" would be reading noise, and the paired no-memory condition exists precisely to make that visible. |
| Session-major candidate pool off by default | Measured on the same machine: MRR +0.019 (p=0.25) against item recall@10 −0.019 (p=0.25) — noise in both directions, and the one thing it clearly does (widen the payload) is the thing the earlier pool 300→800 experiment showed is not the binding constraint. |
| Session-level reranking off | Every metric sits below the entry-level stage (MRR −0.059, p=0.006) and lowering the blend weight drifts monotonically back toward the baseline — the attenuation signature of noise, not signal. A single chunk does not stand for a 96-entry session. |
| `max_evidence_per_session` left at 3 | cap 2 measures item recall@10 +0.0498 (p<0.001), but with zero queries changing session MRR it is a window effect, not a ranking gain. Priced against `run_evidence.py`, every prefix row of cap 3 is ≥ cap 2 and `decisive` falls monotonically as the cap tightens (0.733 → 0.700 → 0.633 → 0.500 at cap 1). The gain is on a proxy of window coverage, the loss on a proxy of usefulness — so the default stays. |
| Listwise reranking off by default | Every setting scored below the no-listwise configuration and cost ~5x search latency. See the table above; the attenuation signature shows it adds noise here, but the proxy measures file overlap rather than usefulness, so this is unresolved rather than settled. |
| Dense enabled by default, device `auto` | The dense cost/benefit flips with hardware (see the table above). `auto` resolves to CUDA when present and CPU otherwise, so one image is fast on a GPU host and still contract-compliant on a CPU one, rather than being tuned for whichever machine happened to measure first. |
| Rerank pool 120 | MRR 0.684 / 0.746 / 0.772 at top_n 30 / 60 / 120: larger is better, and on GPU the 120-pool costs 0.5 s per search, so there is no reason to shrink it. |
| Fixed rerank temperature, not max-normalisation | Normalising by the head's maximum score made every contribution depend on which items happened to be reranked, so changing `rerank_top_n` produced an incoherent sequence (MRR 0.818 → 0.772 → 0.684 as the pool grew). A fixed temperature makes the mapping absolute; the sequence is now monotone (0.684 → 0.746 → 0.772 for top_n 30 → 60 → 120). This also means the earlier 0.8183 figure was an artifact of the flawed normalisation, which is why every number above was re-measured. |
| Clip rerank documents by token count, in one batch | Characters are a bad cost proxy on this model family: 2 048 characters can be 630 tokens while 40 characters is 21, and latency scales with real tokens (4 ms/doc at 21 tokens, 48 ms/doc at 1 034). Token clipping took long-memory reranking from 48 to ~14 ms/doc. Batching the tokenizer call (120 docs in one call rather than 120 calls) took the pool of 120 from 6.0 s to 2.1 s. |
| Keep the noise gate at 0.15 | On this benchmark gate=0 and gate=0.15 score identically, because lexical/entity recall already bounds the candidate set: the gate is not the active constraint here. It is retained because its purpose is the *unrelated-query* case, which this dataset does not exercise — that case is covered by `tests/test_ranking_scale.py` with synthetic same-repo noise. |
| Promote the operative chunk, but only within the top session | Promoting it in every session raised decisive evidence 0.500 → 0.667 and ambiguity 0.367 → 0.567 simultaneously, netting *below* baseline. Restricted to the session we already rank first it gained on all three axes (0.567 / 0.333 / 0.333). The same lever applied everywhere is the same lever applied to evidence we do not believe. |
| Do not cap distinct sessions (`evidence_max_sessions=0`) | The prediction was that tightening the cap buys precision with coverage we could spare, since the answer session was retrieved 100% of the time. Wrong: it is in the payload 100% of the time but is the *top-ranked* session only 70% of the time, so `decidable` stayed flat while decisive fell. What actually needs work is session ranking, not assembly breadth. **Under the prefix metric this is now much less clear** — `max_sessions=1` scores the same `decidable` at 676 tokens instead of 6 518; see the section above. Re-test before treating this as settled. |
| Session-major assembly is free on the retrieval proxy | MRR 0.7800 and recall@10 0.7186 unchanged, nDCG@100 0.7306 → 0.7368, items 67 → 59. File-overlap ground truth cannot distinguish which chunk of a session leads, so a change that only affects chunk *identity* inside a session shows up as no cost. |
| Chunk kind labels tightened against the corpus audit | 95% of `config` chunks were the task prompt and 43% of `diff` chunks had no hunk at all (`scripts/audit_chunk_kinds.py`). Kept on index-correctness grounds: the flat-to-mixed metric movement (recall@10 +0.0028, nDCG@10 −0.0075) is within one query, and the label decides which entity extractors run, which file overlap cannot price. |
| No `code` bonus for debug intent | Operative evidence is concentrated in `code` chunks (2 869 of 3 464), but weighting the kind cost recall@10 (0.7214 → 0.7115), precision@10 (0.2323 → 0.2267) and mean prefix `decidable` (0.408 → 0.383) while only MRR rose. A kind label cannot isolate a 9% minority inside it. |

### What this benchmark cannot tell us

- **It is retrieval-only.** There is no Answer/Eval stage here, so it cannot show
  whether a returned memory helps the answer model. That requires the P3
  end-to-end harness.
- **The relevance definition may not match the organisers'.** If CAMBench credits
  a session for a shared *technique* rather than shared *files*, our lower-bound
  ground truth understates performance.
- **It is not the scored suite.** Never present these numbers as official.
