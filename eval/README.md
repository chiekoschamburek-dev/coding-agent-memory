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
session into ~95.6 memory entries, and `assemble` then capped a session at 3 of
them (the default when this was measured), so the observed packing was **2.06
memory entries per session**. That meant `top_k=100` bought roughly **48
sessions, not 100** — and with no per-session cap it would have bought one or
two, since a single session alone holds enough entries to fill the whole
payload. (The defaults have since moved to cap 5 with at most 2 sessions per
payload; see the multi-span section below.)

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
configuration at the time (cap 3, sessions uncapped), with entry
`recall@10 = 0.5823` (n=89, 333 relevant sessions):

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
thinner than `top_k` suggests** — 59.2 entries over 28.6 sessions under those
defaults, because the 0.15 noise gate ended assembly long before the 100-item
budget was reached (`evidence.py` breaks on
`group[0].final < min_evidence_score`, not on `len(items) >= top_k`). Under the
current defaults the shape is different: at most 2 sessions × cap 5 items, so
the session caps — not the gate — bound the payload (mean 2 434 tokens on the
30-question evidence set; see the multi-span section below).

**The cap is a mechanical lever, not a ranking effect.** The session set is
gate-decided, so lowering `max_evidence_per_session` costs no retrieval — it only
changes how many sessions the first k slots can span:

| cap | item recall@10 | sessions spanned by 10 slots | equals |
|---|---|---|---|
| 3 (the default when measured) | 0.5823 | 4.00 | session recall@4 = 0.5818 |
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
| **3 (default when measured)** | 67.2 | 6548 | 0.433 | 0.433 | **0.433** | **0.367** | **0.333** | **0.700** |
| 4 | 75.1 | 6996 | 0.433 | 0.400 | 0.400 | 0.367 | 0.400 | 0.733 |

cap 1 is a clear loss — one span per session drops decisive evidence from 0.700
to 0.500, which is the failure README already suspected. Between 2, 3 and 4 the
differences are one or two questions out of thirty, i.e. inside the noise, but
every prefix row of cap 3 is ≥ cap 2, and `decisive` falls monotonically as the
cap tightens (0.733 → 0.700 → 0.633 → 0.500).

So cap 2 buys +0.0498 on a *proxy of window coverage* and pays for it on the
proxy of *usefulness*, and the payment is on the axis the platform actually
scores. It is not adopted; the default stays at 3. **This stand-off is what the
2026-09-25 multi-span change resolved by moving the cap *up* to 5 while capping
distinct sessions at 2 — see that section below; the rows above are the
pre-change record.**

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

**Payload-shape comparison, 8 repeats per question (2026-10-02).** The pure-RAG
baseline (see the baseline section above) settles the question the evidence
metric left open: does the 44 k-token "return everything" payload out-answer
the 2.4 k-token denoised one once a real answer model reads both? All three
arms at n=90, 8 answers each, majority-voted (`scripts/compare_e2e.py`):

| condition | majority acc | per-pass range | unanimous | entries shown | relevant sessions in context |
|---|---|---|---|---|---|
| no memory (prior) | 0.700 | 0.700–0.722 | 97.8 % | 0 | — |
| **codemem (shipped assembler)** | **0.711** | 0.711–0.711 | 100 % | 5.8 | 1.01 |
| pure RAG (dense top-100, whole chunks) | 0.689 | 0.689–0.700 | 98.9 % | 99.2 | 2.70 |

Paired exact McNemar: codemem vs RAG +4/−2 (p=0.69, 95 % CI on the delta
[−0.078, +0.033]); codemem vs prior +2/−1 (p=1.0); RAG vs prior +3/−4 (p=1.0).
The answer model's noise floor is small on this question type (near-unanimous
across 8 identical calls), so this null is measured, not sampled.

**The reading that matters: RAG delivered 2.7× more relevant sessions into the
context and 17× more tokens, and answered no better than a model given
nothing.** The evidence metric's ambiguity collapse (RAG decidable 0.333) did
**not** materialise as accuracy loss — gpt-4o-mini simply ignores the 44 k
distractors on a question type where the answer is a file name it can match
against the options. Symmetrically, codemem's denoising bought nothing
measurable either: with the answer session present in only ~1 relevant session
per question, accuracy already sits at the prior's ceiling for this type.
Both payload shapes are, on file-localisation multiple choice, dead weight
around the model's own prior — which is consistent with the earlier case
study above and is now replicated at 8× the sampling.

**Where the two payload shapes can still separate:** a question type the prior
cannot answer from its own knowledge — the procedure/claim questions
(`eval/build_qa_procedure.py`) whose answers exist only inside a recorded
session — and a corpus scale where the platform's 117 760-token input budget
starts binding the 44 k payload. The procedure comparison has since been run;
see the next subsection of the Answer-evaluation section.

**Procedure questions, where the answer exists only in memory (2026-10-03).**
`qa_procedure.json` asks: given the issue, which of four verbatim transcript
excerpts came from the session that handled it? The gold excerpt names no file
the issue names, so pretraining prior should not carry it — the answer exists
only in the stored sessions. Same three arms, n=32, 8 answers each:

| condition | majority acc | per-pass range | entries shown | answer session in context |
|---|---|---|---|---|
| no memory (prior) | 0.406 | 0.406–0.438 | 0 | — |
| **codemem (shipped assembler)** | **0.469** | 0.438–0.469 | 5.9 | 43.8 % |
| pure RAG (dense top-100, whole chunks) | 0.312 | 0.281–0.344 | 99.2 | **84.4 %** |

Paired exact McNemar: **codemem vs RAG +6/−1, p=0.125 two-sided (0.0625
one-sided), delta +0.156 with 95 % CI [0.000, +0.312]**; codemem vs prior
+4/−2 (p=0.69); RAG vs prior **+0/−3 (p=0.25, delta −0.094, CI upper bound
exactly 0.000)** — the only arm of the whole investigation to land below the
no-memory prior.

Three readings, each measured rather than inferred:

1. **The noise penalty is real where it can bind.** RAG retrieved the answer
   session **84.4 %** of the time against codemem's 43.8 % — nearly twice the
   recall — and answered **worse than a model given nothing**. Buried in 99
   same-repository transcripts, the right excerpt competes with dozens of
   maximally confusable ones; retrieval volume was not value. This is the
   evidence metric's ambiguity mechanism showing up in actual answers, which
   it refused to do on file localisation.
2. **The prior floor is higher than the design intended.** no-memory scores
   0.406, not 0.25: the excerpt options themselves leak signal (the model
   matches the issue text against the excerpts' content), so this question
   type measures "identify the session" with a 0.406 head start. The
   separators between arms are therefore compressed; the true memory
   contribution on a leak-free variant would be larger.
3. **codemem's own ceiling on this type is retrieval-bound.** The answer
   session reached the context only 43.8 % of the time — the issue-phrase
   queries do not surface the claiming session through lexical/entity
   channels, exactly the failure the L3 card design targets. RAG's dense
   channel finds it easily; what it cannot do is make the payload survivable.
   The two arms fail on different axes, which is why the combination —
   dense-recall reach with gated, denoised assembly — is where the remaining
   headroom sits.

n=32 and a +6/−1 split sit below the project's two-sided p<0.05 bar, so this
is recorded as a directional result with a mechanism that explains it — not as
an established effect. It is nonetheless the first measurement in this file
where the two payload shapes separated, and it separated on the axis the
evidence metric predicted.

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

**Selecting the cross-encoder's window instead of truncating it: measured, no
effect.** The listwise experiment above found that a judge scores better with a
longer excerpt, which suggested the cross-encoder was reading the wrong text:
it gets `text[:rerank_max_chars]`, i.e. the *opening* of a memory rather than
the part that matches. `rerank_span_tokens` replaces that prefix with the
query-term-density window `_select_span` already uses for returned content.

| | MRR | nDCG@10 | recall@10 | item MRR | item recall@10 |
|---|---|---|---|---|---|
| prefix (default) | 0.7922 | 0.6787 | 0.7344 | 0.7424 | 0.5823 |
| span, 200 tokens | 0.7850 | 0.6753 | 0.7277 | 0.7329 | 0.5869 |

Paired bootstrap: MRR −0.0071 (p=0.63), item recall@10 +0.0047 (p=0.58),
session recall@10 −0.0067 (p=0.50) — and only 7–10 of 89 queries change at all.
That is the explanation, not just the verdict: on this corpus a memory entry is
a median of 151 characters, so `rerank_max_chars` 2000 already covers most of
them whole and there is no wrong-window problem to fix, while a 200-token
selection *shortens* the long ones. It also costs latency (`_select_span` runs
per candidate; search mean 2.20 s against 1.49 s). Left off by default.

**The window is a property of the checkpoint: the equal-pool experiment
(C arms, 2026-09-26/27).** The section above measured *where* the cross-encoder
reads inside a memory; this one measures *how much* it reads, and it is where
`rerank_max_length` / `rerank_doc_tokens` stopped being a single global default.

Setup. Three arms on the same benchmark and evidence harness, pool 120
throughout. **M** is the shipped configuration: `cross-encoder/ms-marco-MiniLM-L-6-v2`
with the MiniLM-era window (pair cap 512, document budget 200). **C2** and
**C1** are `BAAI/bge-reranker-v2-m3` (a checkpoint with 8 194 learned
positions) at two windows — C2 at the same 512/200, C1 at 2048/800 — so C2 vs
M isolates the model at a fixed window and C1 vs C2 isolates the window at a
fixed model. C1/C2 ran back-to-back from one script (`.equal_pool.sh`) with
every other knob pinned (`--rerank-probability-scores`, cap 5, 2 sessions,
item tokens 800, position weight 1.0, promotion 2). An earlier same-evening
pair (arms B1/B2) had already contrasted the two models/windows on GPU, but at
the time the token-window plumbing did not exist at all — no
`--rerank-max-length` / `--rerank-doc-tokens` flags and no env vars for them
(verified against the committed tree; both were written *after* those runs) —
so any window contrast it measured went through the older character clip
(`rerank_max_chars`). B is recorded below but not trusted for the decision.

| arm | reranker | pair cap / doc budget | MRR | item MRR | entry nDCG@10 | decidable (n=30) | decisive | ambiguous | payload tokens | search mean |
|---|---|---|---|---|---|---|---|---|---|---|
| M (shipped) | MiniLM-L-6-v2 | 512 / 200 | 0.7584 | 0.7182 | 0.4845 | 0.567 | 0.767 | 0.300 | 2 408 | 0.72 s |
| C2 | bge-reranker-v2-m3 | 512 / 200 | 0.7472 | 0.6966 | 0.4685 | 0.400 | 0.667 | 0.333 | 2 535 | 8.1 s |
| C1 | bge-reranker-v2-m3 | 2048 / 800 | 0.7697 | 0.7416 | 0.4933 | **0.633** | **0.767** | **0.200** | 3 234 | 234 s |

Three paired readings:

- **C2 vs M — swapping the model alone is a regression.** decidable 0.567 →
  0.400 (−8/+3 questions), item MRR 0.7182 → 0.6966. The 512/200 window was
  sized for MiniLM, whose 512 learned positions make it a natural fit; the
  same numbers handed to bge-reranker-v2-m3 clip 76.7 % of Edit/Write/MultiEdit
  memories above 200 tokens — exactly the diff hunk or tool call the judge is
  being asked to score. The judge does not get worse; it is fed less.
- **C1 vs C2 — the window is the whole effect.** decidable 0.400 → 0.633
  (+9/−2, exact McNemar p=0.065, one-sided 0.033), decisive present
  0.667 → 0.767, ambiguity 0.333 → 0.200, entry nDCG@10 +0.0248 (p=0.074),
  item MRR +0.0449 (p=0.13), payload +0.7k tokens. The proxy metrics move in
  the same direction but under-claim the effect (recall@10 +0.012, p=0.55):
  recall is labelled per *session*, and a bigger window rarely changes whether
  a session is found — it changes whether the session's operative chunk
  survives the judge's reading of it. This is the same
  session-vs-entry blindness documented for the position tilt below.
- **C1 vs M — the shipped configuration, challenged and held.** decidable
  0.633 vs 0.567 (+6/−4, p=0.75), decisive identical at 0.767, ambiguity
  0.300 → 0.200. No single comparison is significant at n=30, but every axis
  points the same way, and the mechanism — the window binding on long
  operative memories — is measured directly in the corpus, not inferred.

**Why the average effect is small and the evidence effect is not.** The median
memory entry is 151 characters, so any window ≥ ~600 tokens covers it whole;
raising the budget cannot help the median document and does not. What it
changes is the tail that carries the evidence: 24.5 % of memories exceed 200
tokens, and among the Edit/Write/MultiEdit ones 76.7 % do. The questions the
evidence harness asks are precisely questions about that tail — "does the
payload show the edit for the file in question" — so a window that decides the
tail decides the metric. The retrieval proxy averages over all 333 pairs and
dilutes exactly the subset that moved.

**Latency prices the window, and the price is hardware-shaped.** On the CPU
the C series ran on (see below), C1 cost 234 s per search against C2's 8.1 s —
a 29× multiplier for 4× the tokens, which is attention's quadratic growth plus
more documents actually reaching the budget. On GPU the same window contrast
measured ~3 s per search (B1) against ~1 s (B2). This is why the decision
ships as *checkpoint-keyed defaults* rather than a bigger constant: a
checkpoint that can read 8 194 positions defaults to 2048/800, MiniLM-era
checkpoints keep 512/200, an explicit env/CLI value always wins, and a
CPU-only deployment can still pin the small window back.

**Provenance caveat, recorded because it bounds the claim.** The C series ran
overnight at ~8–234 s per search; the B series, on the same configurations
earlier the same evening, ran at ~1–3 s per search. The most likely reading is
that B had the GPU and C fell back to CPU (the B/C per-query outputs agree
closely — B1 vs C1 identical sequences 60/89, first-rank differences 4/89 —
which is what a precision change, not a configuration change, produces), but
this is inference from latency, not a logged fact: the reranker's
`pair_tokens`/`doc_tokens` startup logging was added *because* this run made
the mismatch invisible. Two consequences. First, the C1-vs-C2 window contrast
is internally clean (same series, same device, everything pinned) and is the
basis of the decision. Second, B's evidence-side reading of the same contrast
(decidable 0.500 vs 0.467, flat) is not, and is superseded; its proxy-side
reading (11/89 queries moved, MRR 0.747 → 0.764) agrees with C in direction.

**Decision.** `rerank_max_length` and `rerank_doc_tokens` (and
`rerank_probability_scores`, for the same one-time-surprise reason) resolve
with the checkpoint when unset: bge-reranker-v2-m3 → 2048/800/probability
scores on, everything else → 512/200/off. An end-to-end Answer pricing of the
+0.7k payload tokens is still owed, as is a GPU re-run of C1 before quoting
its latency to the platform.

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

### Baseline: pure RAG on the same index (2026-10-02)

`scripts/baseline_dense_rag.py`. The question this baseline settles: how much
does the memory machinery actually buy over the canonical RAG shape — embed the
query, cosine top-k, return chunk text verbatim? The baseline patches
`SearchService.handle` at runtime, adds memories through the ordinary `/add`
path and reads the vectors Add already stored, so it shares the index (same
chunks, same embeddings) with codemem and differs *only* in retrieval and
assembly: no query planning, no channels, no identifier scoring, no fusion, no
noise gate, no session quotas, no reranker.

**On the file-overlap proxy, pure RAG beats the full machinery on every
metric** (paired, 89 queries, RAG → codemem): session recall@10 0.7506 →
0.4746 (Δ −0.276, p<0.001), session nDCG@10 0.7142 → 0.5230, entry MRR
0.8141 → 0.7182, and even entry precision@10 0.6955 → 0.5361 — at a payload of
100 entries spanning 21.6 sessions against codemem's 9.2 entries over 2. On
the stricter shared-≥2-files labels the gap holds (session recall@10 0.880 vs
0.660, −0.219, p<0.001), so this is not weak-label dilution. Two things
produce it, and neither is a retrieval failure of our channels (the funnel
shows 94.3 % of relevant sessions reach the pool): the **2-session budget**
throws away everything ranked third or below — 16/89 queries emit no relevant
session at all — and cosine similarity is genuinely good at this label, since
sessions touching the same files share the vocabulary the query embeds near.

**On the evidence metric, the ordering inverts — and the mechanism is
ambiguity, not truncation** (30 questions, paired). The official API Guide's
Runtime Rules cap the Answer stage at a 128 000-token shared window (8 192
output + 2 048 safety reserved, **117 760 input tokens**), and state that
"*if needed*, Answer keeps a token-counted prefix of Search candidates in
returned rank order". At this corpus's scale neither payload approaches that
budget — codemem returns ~2.4 k tokens, pure RAG ~44 k — so the prefix rule
likely never fires here and the platform's operating point is the
full-payload row below. The prefix rows are kept as a stress model of a
tighter budget:

| payload | RAG decidable | codemem decidable | RAG tokens | codemem tokens |
|---|---|---|---|---|
| @1 000 (stress) | 0.100 | 0.633 | 1 000 | 1 000 |
| @2 000 (stress) | 0.400 | 0.600 | 2 000 | 1 860 |
| @4 000 (stress) | 0.533 | 0.567 | 4 000 | 2 394 |
| @8 000 (stress) | 0.567 | 0.567 | 8 000 | 2 408 |
| **all (~the platform's operating point)** | **0.333** | **0.567** | 44 335 | 2 408 |

The full-payload row is the mechanism in its purest form: RAG's 44 k-token
payload contains the decisive evidence **93.3 %** of the time (codemem 76.7 %)
— returning everything does find it — but a contradicting distractor appears
for **63.3 %** of questions against codemem's 30.0 % (paired −10/0, p=0.0020),
so decidable collapses to 0.333. With the whole payload visible, the answer
model drowns rather than starves. Codemem's denoised 2.4 k-token payload holds
0.567 at every budget, and would keep that property under a corpus large
enough for the platform's prefix rule to actually bind — where the stress rows
say it opens a 6× decidable lead at 1 000 tokens. The machinery's value is
ambiguity control first, prefix efficiency second; its price is recall on a
session-generous label plus ~0.6 s per search (146 ms for RAG, 724 ms for
codemem, the reranker's share).

**What this changes in how our own numbers are read.** The proxy's
recall@10 0.4746 is a *payload-shape* number, not a retrieval number: lifting
the session budget to 6 measured 0.6624 (precision@10 0.52 → 0.31) in the
funnel work, still below RAG's 0.7506 on this label while precision stays
above it. Quoting recall against a RAG baseline therefore requires quoting the
payload sizes next to it — and since the platform's 117 760-token input budget
admits even the 44 k-token payload whole, the consumer the small payload is
built for is one that must not be drowned in same-repo contradictions. The
end-to-end comparison has since been run (see the payload-shape comparison in
the Answer-evaluation section): on file-localisation multiple choice neither
payload shape separates from the no-memory prior — the discrimination now
waits on harder question types or a budget-binding corpus scale.

### Experience cards: implemented, measured, inert as shipped (2026-10-03)

The L3 card is the one structural remedy the funnel work left standing: a
session enters ranking as a single comparable object (the cross-encoder reads
(query, overview) as one pair, which no single chunk of a ~100-chunk session
can stand in for — the session head names the task file only 12.9 % of the
time) and recall gains a candidate that is not one of ~100 same-session
siblings. Implemented per the card schema (docs/DESIGN.md): one generated
overview per session during Add, cached by prompt content hash, embedded for
dense, `kind='card'` in the memory table; it flows through recall, scoring and
reranking like any row, but is never emitted — `data[].content` stays a
verbatim chunk span — and a session whose only member is its card emits
nothing (`CODEMEM_CARDS`, off by default; 7 invariant tests in
`tests/test_card.py`).

Measured on all three harnesses, against the shipped configuration:

| harness | shipped | cards on | paired |
|---|---|---|---|
| proxy, session recall@10 (89 q) | 0.4746 | 0.4784 | +0.004, p=0.77, **2/89 queries moved** |
| proxy, session MRR | 0.7584 | 0.7584 | identical to four decimals |
| evidence, decidable (30 q) | 0.567 | 0.567 | +1/−1, p=1.0 |
| evidence, ambiguous | 0.300 | **0.233** | +2/−0 improved |
| evidence, payload | 9.6 entries / 2 408 tok | 5.9 / 1 883 | **−22 % tokens** |
| procedure e2e, accuracy (32 q × 8) | 0.469 | 0.438 | −1 question, p=1.0 |
| procedure e2e, **answer session shown** | 43.8 % | **43.8 %** | unchanged |

The costs are not free: Add went from 180 s to 1 652 s for the benchmark
corpus (300 LLM calls, serialized at ~5.4 s each) and search latency from
724 ms to 1 736 ms. The overview quality itself is good — the generated
summaries name concrete files, the bug, the approach and the outcome — so the
null is not a generation-quality failure.

**Why it is inert: the recall promise is structurally disabled by the safety
invariant.** The card can only *re-rank* sessions that already have an
admitted chunk, because "a card qualifies nothing" (card invariant 5): a
session whose chunks never cleared the informative-channel gate has no
emittable members even when its card matches the query perfectly. The 56 % of
procedure questions where the answer session is missing are exactly those —
so the card, as shipped, cannot touch the number it was designed to move, and
answer_session_shown did not move by a single question. What survived is a
mild denoising side effect: cards displace weak chunks in the pool, so the
payload shrinks 22 % and ambiguity drops, at nine times the Add cost.

**The fork this measurement forces.** Either (a) relax invariant 5 — let a
card hit carry its session's *span chunks* into the payload (expansion pulls
verbatim chunks by span, so returned text stays compliant; the noise gate
then has to be re-derived for card-admitted sessions, whose admission rests
on generated text), and re-measure the procedure e2e, where the headroom is
measured at +0.156 over pure RAG; or (b) retire the card lever and accept the
43.8 % answer-session ceiling on procedure questions. As shipped, (b) is what
the numbers say: everything the card adds, it adds to sessions that were
already findable.

**Fork (a), implemented and measured: still inert, and the diagnosis moves
upstream (2026-10-03).** `card_expansion` (opt-in, `CODEMEM_CARD_EXPANSION`)
vouches a gated card's session-tail chunks into the payload — verbatim spans,
position-prior selection, capped, budgeted; 3 more invariant tests. Armed
with all 300 cards written and embedded (verified in the run's database), on
both harnesses it moved **nothing**: proxy — cards-only rows already showed
2/89 queries moving, expansion adds none; evidence — decidable 0.567 = 0.567;
procedure e2e 0.469 → 0.438 (−1 question, p=1.0), answer_session_shown
43.8 % → 43.8 %, not one question changed. The mechanism is now closed: a
card that only dense found is a *dense-only* candidate, and the evidence gate
(`INFORMATIVE_CHANNELS`) discards it before ranking — so expansion's trigger
("a session whose only admitted member is its card") never fires for exactly
the sessions it was built for. The relaxation was necessary but not
sufficient; the binding constraint sits one stage earlier, at the gate.

**The gate lever, measured as a dose-response (dense-only admission).**
`dense_eligible` admits a dense-only candidate above an absolute cosine floor
(0.45 default, strength = cosine above floor) — a switch that already existed
and had never been measured end-to-end. Procedure e2e, n=32 × 8:

| floor | majority acc | vs baseline | answer session shown |
|---|---|---|---|
| off (baseline) | 0.469 | — | 43.8 % |
| **0.45** | **0.500** | +1/−0 | 43.8 % |
| 0.30 | 0.469 | +1/−1 | 40.6 % |

The +1 at 0.45 is a verified mechanism case, not drift: `django-27995` was
8/8 wrong under the baseline, and with dense-only candidates in the pool the
payload composition changed and 7/8 votes moved to the gold answer (flag
confirmed live — votes differ, payload sizes do not). At 0.30 the same
question stays fixed but `sympy-13039` breaks the other way: weak dense
candidates crowd the answer session out of its payload (shown True → False)
— the noise effect returning through the opened gate. Costs elsewhere are
within noise: proxy at 0.45 moves 1/89 queries (session MRR −0.011),
precision@10 −0.015 (p=0.09); evidence at 0.45 halves ambiguity (0.300 →
0.167) while losing the same amount of decisive markers — decidable flat.
One harness determinism check for free: an accidental unflagged rerun of the
proxy reproduced every metric to the digit.

Where this leaves the card programme: the complete cell — cards + expansion
+ gate 0.45 — has now been run: **0.469**, exactly baseline (paired +1/−1 vs
baseline, −1/0 vs the gate alone). The card adds nothing on top of the gate;
the gate alone remains the best measured arm. Full procedure-e2e leaderboard
(n=32 × 8, majority vote): RAG 0.312 < cards 0.438 = cards+expansion 0.438 <
baseline 0.469 (reproduced exactly on an inadvertent rerun) = cards+expansion
+gate 0.469 < **gate 0.45 alone 0.500**. The honest summary of this arc:
**the 43.8 % answer-session ceiling on procedure questions survived every
card variant tried, in every combination**; the only lever that moved the
axis at all is plain `dense_eligible@0.45` (+1/32, p=1.0 — recorded, not
shipped: one question does not clear the project's two-sided p<0.05 bar),
and raising n on the procedure set is the cheapest way to make any of these
one-question effects decidable.

### Large-set re-measurement: the question type now separates arms (58 questions)

`build_qa_procedure.py --per-query 6` raises the yield from 32 to 58 — the
corpus's structural ceiling, not a filter artefact: of the candidate
assistant messages, 14 713 are tool-call records and 6 030 are outside the
claim window, so prose-with-a-recorded-cause anchored to a scored issue is
what runs out. The adversarial anti-guessability construction is kept intact
(issue-word baseline 0.000, zero gold 6-grams in the question, gold positions
balanced). Four arms, n=58 × 5 repeats, majority vote:

| arm | majority acc | vs no-memory | vs shipped |
|---|---|---|---|
| no memory (prior) | 0.328 | — | — |
| shipped codemem | 0.362 | +5/−3, p=0.73 | — |
| **shipped + dense_eligible@0.45** | **0.397** | +7/−3, p=0.34 | +2/−0, p=0.50 |
| pure RAG (dense top-100) | **0.224** | +3/−9, p=0.15 | — |

Two results at n=58 that the 32-question set could only hint at:

1. **The first statistically significant separation of the programme: the
   gated pipeline beats pure RAG at p=0.0063** (+1/−11, +0.172, CI [−0.293,
   −0.069]). RAG reaches the answer session 89.7 % of the time against our
   34.5 % — and answers **below chance** (0.224 against 0.25). Same dense
   channel, same questions, same answer model: the difference is the gate
   and the assembly. "Return everything" is now not merely worse, it is
   *actively harmful* on the question type whose answers exist only in
   memory, and the effect size (−17 pp) is the noise mechanism measured at
   scale.
2. **`dense_eligible@0.45` holds its sign on the larger instrument** (+2/−0
   over shipped, +0.069 over prior) — the only lever with a consistent
   direction on both set sizes and a verified mechanism case behind it.
   Still short of significance at n=58; it becomes the default candidate
   if the effect survives the next instrument revision.

What did not move: `answer_session_shown` sits at 34.5 % on this set for
both shipped and gate arms — the newer questions anchor on sessions our
channels reach even less often, so the recall ceiling is the same wall, seen
from further away. The memory-over-prior gain is positive-directional but
small (+0.034 shipped, +0.069 gated); on this question type, controlling
noise is worth more than adding reach — which is the whole thesis of the
small payload, now measured against a significant reference point.

### The session-score estimator: top-k mass, the first ranking lever that works (2026-10-04)

`session_score_topk`. The session's score was the **max** over its admitted
members — the estimator the attribution work called wrong (a session's head
chunk names the task file only 12.9 % of the time). `session_score_topk=k`
replaces it with the sum of the top-k member scores; the noise gate keeps
reading the head. Three instantiations of "rank sessions directly" had
already failed (session-representative rerank MRR −0.059 p=0.006, listwise
judge below baseline, cards gated out) — all of them replaced the
*representative object*; this one only fixes the *aggregator*, and it is the
first session-ranking change with a significant win:

| metric (proxy, 89 q, paired) | max (shipped) | top-2 | top-3 |
|---|---|---|---|
| session recall@10 | 0.4746 | **0.4935** (p=0.004) | 0.5004 (p=0.045) |
| session nDCG@10 | 0.5230 | **0.5429** (p=0.005) | 0.5470 (p=0.044) |
| entry recall@10 | 0.4746 | **0.4935** (p=0.004) | 0.5004 (p=0.045) |
| entry nDCG@10 | 0.4845 | 0.4997 (p=0.011) | 0.5042 (p=0.081) |
| session MRR | 0.7584 | 0.7809 (p=0.13) | 0.7865 (p=0.16) |
| entry precision@10 | 0.5361 | 0.5587 (p=0.076) | 0.5490 (p=0.45) |
| payload entries | 9.25 | 9.66 | 9.78 |

**Top-2 is the better arm**: the recall/nDCG gains are significant, precision
does not degrade, and the payload grows by 0.4 entries (top-3 buys +0.3pp
recall for +0.11 more entries with visibly weaker precision confidence).
Evidence metric at top-2 (30 q): decidable unchanged (0.567), decisive
0.767 → 0.800, payload −16 % (2 408 → 2 014 tokens) — no downside measured.
Procedure e2e (58 q × 5, after the relay quota was topped up; two earlier
attempts were invalidated wholesale by `insufficient_user_quota` 403s and
are not results): **0.379 vs shipped 0.362** (+1/−0, p=1.0), answer-session
reach 34.5 % → 37.9 %. The accuracy gain is one question — within the noise
this instrument can resolve — but the direction agrees with the proxy's
significant recall/nDCG gains, and the cost axis (payload, ambiguity,
decisive markers) shows no regression on any instrument. Shipped as
`session_score_topk=2`? Not yet: the e2e gain is one question, and the
proxy's significant recall gain is on session-labelled ground truth that
cannot price payload usefulness — the same asymmetry documented for the
entry-order rejection. Default stays 1; top-2 is recorded as the best
measured ranking candidate alongside `dense_eligible@0.45`.

**The combination cell (gate 0.45 + top-2) was measured and the levers do
not stack on answer accuracy.** Proxy (89 q): the combination is the best
ranking configuration on record — session recall@10 0.5013 (p=0.012 vs
baseline), and significantly above the gate alone (session MRR +0.034,
p=0.003; entry recall +0.028, p=0.003), with precision unchanged. Evidence
metric (30 q): decidable flat, payload −12 %. Procedure e2e (58 q × 5):
combo 0.379 = top-2 alone, −1 question vs the gate alone (p=1.0 everywhere);
answer-session reach 37.9 %. Leaderboard on the memory-dependent instrument:
RAG 0.224 < prior 0.328 < shipped 0.362 < top-2 = combo 0.379 < **gate alone
0.397** — every gap ≤ 2 questions. The two levers act on different stages
(admission vs ordering) and stack measurably on the file-overlap proxy, but
on answer accuracy they are within noise of one another; the ship decision
between {gate, top-2, combo} stays open pending a larger instrument, and the
default configuration stays unchanged.

### Session-feature fusion: the first significant session-ranking win (2026-10-04)

Path 1 of the three-road plan (free structural signals, offline replay
`scripts/session_features.py`). The replay validated bitwise against the
measured pipeline once two aggregations were separated — pair-micro 0.2763
(the funnel's 27.6 %) vs query-macro 0.4746 (the recall@10 run_benchmark
reports) — and exposed a mechanism worth recording: the naive "first two
sessions in rank order" loses 20 points to the real assembler, because the
noise gate **skips** weak-head sessions and later ones take the slot. Four
features were replayed; two survived, two were rejected on the replay itself:

| feature | replay (macro) | verdict |
|---|---|---|
| F1 query ↔ session's **first message** cosine | 0.4747 alone; **0.5040–0.5046** rank-fused with the head | kept — the issue statement lives at the trajectory head, in query vocabulary |
| F2 **rare-vocabulary union coverage** of pooled chunks | 0.5001–0.5102 fused | kept — invisible to per-chunk max |
| F3 cause-lexicon prose × query identifier | exactly 0.4746 | rejected — the signal is too sparse in this corpus |
| F4 supersede-chain terminality (timestamp proxy) | 0.1161 alone, negative fused | rejected — strongly harmful |
| F1 + F2 stacked (w=2/2) | **0.5354 offline** | shipped behind the flag |

Implemented as `session_feature_fusion` (off by default): the service computes
F1 (one embed call over the candidate sessions' first messages) and F2 (union
coverage of the query's rare terms by pooled chunk texts), and `assemble`
rank-fuses them into the session order (weights 2.0/2.0, the offline plateau's
interior). Measured in product on all three instruments:

| instrument | shipped | fusion | paired |
|---|---|---|---|
| proxy, session recall@10 (89 q) | 0.4746 | **0.5354** | **+0.061, p=0.001** — matches the offline prediction to four decimals |
| proxy, session MRR | 0.7584 | 0.8483 | +0.090, p=0.009 |
| proxy, entry precision@10 | 0.5361 | 0.5994 | +0.063, p=0.033 |
| evidence, decidable (30 q) | 0.567 | 0.600 | +4/−3, n.s.; payload −8 % |
| procedure e2e accuracy (58 q × 5) | 0.362 | 0.345 | −1 question, p=1.0 |
| procedure e2e, **answer session shown** | 34.5 % | **46.6 %** | **+12.1 pp — the largest reach gain measured** |

The retrieval win is real, significant, and free (no LLM in the graded path,
one embed call per search); the accuracy on the memory-dependent question
type is flat — reaching more answer sessions did not convert into answers,
which is precisely the question path 2 (LLM two-stage selection) is designed
to test, now with better material to select from.

### Path 2: LLM two-stage session selection — the best answer accuracy measured (2026-10-04)

`session_select_llm`. The gate and the budget are untouched; what changes is
**who decides the top-2**: the top ~8 candidate sessions are summarised
(first message + files + two top chunks) and one `gpt-4o-mini` call picks the
two that record the cause or fix — the same judgement the platform's Answer
model makes, made over session-level summaries. This is not the failed
listwise stage: that scored single chunks with no session boundary in view;
this compares sessions as units, and its preference is structurally aligned
with the graded metric. It takes precedence over feature fusion when it
succeeds and degrades to the shipped order on any relay failure (zero
failures across 89 searches + 290 answer calls).

| instrument | shipped | select-llm | paired |
|---|---|---|---|
| proxy, session MRR (89 q) | 0.7584 | **0.8652** | **+0.107, p<0.001** — the largest MRR gain measured |
| proxy, session recall@10 | 0.4746 | 0.5171 | +0.043, p=0.035 |
| proxy, entry precision@10 | 0.5361 | 0.5951 | +0.059, p=0.039 |
| procedure e2e accuracy (58 q × 5) | 0.362 | **0.414** | **+4/−1, +0.052, p=0.375** |
| procedure e2e, answer session shown | 34.5 % | 46.6 % | +12.1 pp (same reach as fusion) |

Procedure-e2e leaderboard after all three roads: RAG 0.224 < prior 0.328 <
shipped 0.362 ≈ feature-fusion 0.345 < gate 0.397 < **select-llm 0.414**.
Both ranking levers are significant on the proxy; the selection stage is the
only one that also moves answer accuracy (+0.052, direction consistent with
its proxy gains, not significant at n=58). Costs: one LLM call per search
(~1 s with the relay, the same model the rules fix for the graded path) and
a relay dependency the shipped configuration does not have — the flag stays
off by default, and the deployment decision between {gate 0.45, select-llm,
both} is deferred to the smoke window. Path 3 (the HyDE card campaign) is
**affirmed as worth doing by path 2's result** — the LLM demonstrably picks
the right sessions from summaries — but deferred: path 2 already delivers
the judgement per-query at Search time, and the card's remaining advantage
(pushing the judgement to Add time permanently, no Search-side relay) does
not outweigh a 30-minute LLM campaign per instrument run before deployment.

### Why select-llm still missed 53.4%: the zero-relay decomposition (2026-10-04)

`scripts/diagnose_select_miss.py`. Before any stacking run, the 31 misses of
the select-llm arm were replayed offline (one Add, one local embed pass, zero
relay calls; the arm's per-question reach joins from the recorded dump):

| bucket | n | meaning |
|---|---|---|
| **shortlist-bottleneck** | **20 (65 %)** | the answer session was outside the top-8 the LLM saw |
| gate-blocked | 0 | no shortlisted session was ever gate-killed |
| llm-judgment | 11 | shortlisted and admissible, not picked |

Three decisions fall straight out:

1. **gate + select stacking is dead**: with zero gate-blocked misses, the
   gate cannot rescue any select-llm miss directly; its only path would be
   reshuffling shortlist membership, which is exactly what fusion already
   does measurably better.
2. **fusion is not redundant — it is the shortlist feeder.** The identical
   +12.1 pp reach was an aggregation coincidence: fusion fixes 6 queries
   select misses, select fixes 12 fusion misses, only 2 overlap. Of the 20
   shortlist-bottleneck misses, the answer session surfaces in the fusion
   payload for 6 (all admissible) — so **fusion-feeding-select has a
   measured ceiling of ~6 questions (+10 pp)**, and a perfect shortlist
   would rescue 10.
3. **the one stacking run worth its 58×5 is fusion + select**, expected
   yield 3–6 questions — at or above the instrument's resolution for the
   first time, with the expected value known before running. The remaining
   11 llm-judgment misses are summary quality — parked-card territory.

### The fusion+select stack: measured, does not convert (2026-10-04)

The one stacking run the miss decomposition licensed (ceiling ~6 questions).
Fusion now feeds the shortlist — the fused ordering is applied to the
reranked list before the LLM call (`fused_session_order`, shared by both call
sites; guard-tested) — and the LLM picks from the fused top-8.

| comparison | accuracy | paired |
|---|---|---|
| shipped | 0.362 | — |
| **stack (fusion → select)** | **0.397** | +3/−1 vs shipped, p=0.63 |
| select-llm alone | **0.414** | stack −1 question, p=1.0 |
| gate alone | 0.397 | identical to stack, +2/−2 |

The 6-question ceiling was an upper bound, not an expectation: fusion's
reordering changed the shortlist (mean relevant-in-context rose 1.03 → 1.24,
payload −0.3 entries) but answer-session reach *fell* slightly (46.6 % →
43.1 %) — the fused shortlist traded some sessions the LLM had been picking
correctly for different ones, and the judgment errors ate the supply. Final
procedure-e2e leaderboard (n=58 × 5, seven arms): RAG 0.224 < prior 0.328 <
fusion 0.345 < shipped 0.362 < gate = stack 0.397 < **select-llm 0.414**.

The session-ranking campaign is measured to completion. Select-llm is the
champion on both the proxy (MRR +0.107, p<0.001) and the answer metric
(+0.052 over shipped, the largest single-arm gain, n.s. at n=58); fusion is
significant on the proxy and the reach leader on its own; neither stack
improves on select alone. All three levers stay flags, defaults unchanged —
the ship decision among {gate 0.45, fusion, select-llm} belongs to the
deployment smoke window, with `docs/SUBMISSION.md` holding the trade-off
tables.

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
| **0** | **top 1 (default then)** | 1.000 | 0.567 | **0.333** | **0.333** | 67.1 |
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

At the defaults then in force (30 questions, cap 3, sessions uncapped; mean
65.7 items / 6 518 tokens returned):

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

**Update (2026-09-25): this stand-off is resolved — see the next section.** The
shipped defaults are now `max_evidence_per_session=5`,
`evidence_max_sessions=2`, and promotion across the top 2 sessions, so the
"leave it uncapped" recommendation above is history, superseded by the
measurement below.

### Confidence-aware multi-span: shorter payload, higher decidable (2026-09-25)

The two linked open decisions — the per-session cap (3 was a stand-off with cap
2) and the distinct-session cap (0, with a recommendation to re-test) — were
revisited together with span selection, and shipped as one change
(`7673cde`):

- **`max_evidence_per_session` 3 → 5, `evidence_max_sessions` 0 → 2.** The
  evidence-sufficiency audit found the decisive chunk often sits deep inside a
  read-heavy trajectory, and a cap of 3 truncated before reaching it. With at
  most two sessions in the payload, the answer session gets up to five slots to
  expose its decisive evidence while distractor sessions stay out of the window.
- **Operative promotion 1 → 2 sessions.** With at most two sessions admitted,
  promoting the operative chunk in both recovers decisive evidence in either
  candidate session without the ambiguity floodgate seen when every session was
  promoted (0.567 at `-1`).
- **Confidence-aware multi-span selection.** Only the top-ranked session's items
  use the multi-span path of `_select_span`: every operative block (an operative
  line plus bounded context) is preserved, then the remaining budget is filled
  with the densest non-operative window. Later sessions keep the single best
  window, so distractor operative lines do not inflate ambiguity. The prose fill
  is skipped: once the operative signal is preserved, padding adds noise without
  raising `decisive_present` (`evidence.py`).

Measured on the 30-question evidence set, same machine, deterministic config
(P1 path):

| payload | mean tokens | decidable @2000 | decidable @all | ambiguity |
|---|---|---|---|---|
| before (cap 3, sessions uncapped, single-span) | 7 190 | 0.567 | 0.400 | 0.433 |
| after (cap 5, sessions 2, multi-span top-1) | **2 434** | **0.567** | **0.500** | **0.333** |

`decidable` at a 2 000-token prefix is unchanged while the payload shrinks 66 %,
and the tail penalty shrinks with it (the gap @2000 → @all narrows from −0.167
to −0.067). That resolves the cap-2 stand-off from the pricing table above: the
window-coverage gain is now had by capping *sessions* instead of *slots per
session*, so the per-session cap is free to move up and buy decisiveness
(0.700 → 0.800 at the same sweep that set `evidence_item_tokens`).

**Caveat carried from the pricing table's footnote:** the retrieval-proxy rows
above this section (MRR 0.79, recall@10 0.73, item recall@10 0.58) were all
measured under the pre-change defaults. The change alters which chunks are
selected and how many sessions appear — a dimension file-overlap ground truth
largely cannot see — but the headline retrieval numbers should be re-derived
with `run_benchmark.py` before they are quoted in any submission material.

### Entry-order emission: measured, rejected on the evidence axis (2026-10-02)

`scripts/exp_entry_order.py`. The shipped assembler emits memories **session by
session**: it walks the globally-scored candidate list grouped by session and
drains up to `max_evidence_per_session` entries from each block. A session's
weaker entries therefore travel ahead of another session's stronger ones. The
hypothesis: emit **entries in global score order**, keeping the per-session cap
merely as a quota, and the ordering the platform actually consumes (a
token-counted prefix of `data[]`) improves.

| arm | walk | per-session quota | entries avg | item nDCG@10 | item MRR | item precision@10 | item recall@10 |
|---|---|---|---|---|---|---|---|
| S0 (shipped) | session blocks | 5 | 9.2 | 0.4845 | 0.7182 | 0.5361 | 0.4746 |
| S1 | global entry order | none | 40.3 | 0.5010 | 0.7428 | 0.6753 | 0.4718 |
| S2 | global entry order | 5 | 9.2 | **0.5035** | **0.7438** | 0.5361 | 0.4746 |
| S3 | global entry order | 1 | 2.0 | 0.5230 | 0.7584 | 0.5225 | 0.4746 |

Recall, fusion, scoring and reranking are shared — every arm walks the same
scored candidate list, and the pairing check confirms all four return the
baseline's session set for 100 % of queries. Two implementation notes recorded
because both produced silently wrong payloads before being caught: the noise
gate must read a session's *best* evidence (gating every candidate ends the
walk on a weak non-head chunk while later sessions' heads would have cleared
it), and the session limit must stop *admission*, not the walk (breaking on the
third session's head strands the admitted sessions' remaining quota). Both bugs
showed up as payload sizes that could not be reconciled with the baseline
(4.1 and 5.2 entries against 9.2).

**S2 vs S0 is a pure ordering comparison over an identical item set** — same
9.2 entries per query, item precision and item recall identical to four
decimals, 0 queries moved — and global order wins: item nDCG@10 +0.0190
[+0.0117, +0.0266], item MRR +0.0257 [+0.0103, +0.0444], both p < 0.001. The
session-level view is invariant by construction (each session's head ranks
above every other candidate of that session, so first-appearance session order
cannot change). The quota axis confirms the cap is load-bearing: S1 floods to
40.3 entries at 20.2 per session and gives back item recall@10 (−0.0028, one
query). S3's nDCG advantage is the already-rejected proxy artifact: at quota 1
each session emits only its head, so the item view collapses into the session
view — the same reason cap=1 was refused in the tuning table (a two-item
payload cannot carry a session's context).

**But S0's block order is not an arbitrary walk — it carries the two
intra-session levers** (position tilt, operative promotion) that the entry walk
deliberately omits to isolate the ordering variable. So the decisive test is
the evidence metric, run with the same entry assembler patched into
`run_evidence.py` (`eval/results/ev_entry_cap5.json`), paired against the
shipped assembler on the same 30 questions:

| metric | shipped (ev_m_M) | entry order | paired |
|---|---|---|---|
| decisive present | 0.767 | 0.433 | 0 gained / 10 lost, p=0.0020 |
| **decidable** | **0.567** | **0.300** | **0 / 8, p=0.0078** |
| decidable @1 000-token prefix | 0.633 | 0.167 | — |
| payload | 9.2 entries / 2 408 tok | 9.6 entries / 2 254 tok | comparable, so this is not a size effect |

Not one question improved. The mechanism is the measured 12.9 % head-is-gold
figure: the operative chunk rarely leads its session on `final` score, so
global score order buries it behind whichever chunk scored highest overall,
while the shipped walk hoists it with promotion and the tilt. The promotion
ablation had already priced those levers at `decidable` +0.267 on a 1 000-token
prefix — an order of magnitude more than the +0.019 nDCG the global walk gains.

**Verdict: the session-major assembler stays.** The ordering gain is real but
it prices the wrong thing: file-overlap labels reward moving *any* entry of a
relevant session earlier, while the answer model needs the *operative* one
readable in the prefix. The hypothesis is not reopened unless someone ports the
position tilt and operative promotion into the global walk and prices that on
`run_evidence.py`. For history: the 2026-09-21 run of this experiment (archived
as `exp_entry_order_cap3_20260921.json`) had entry order winning item
recall@10 by +0.064 under the cap-3, promotion-less assembler of the time —
that diagnosis was correct, and the 2026-09-25 multi-span redesign absorbed
it. Re-tested on the cured assembler, nothing is left to collect on the axis
that decides answers.

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
tried for ranking. (It has since been tried, both bare and crossed with identifier
evidence: see *Two attempts to move the 12.9%* below.)

### Where the decisive chunk is lost, stage by stage

Every number above is labelled per **session**, which asks "did any chunk of the
right session make the window". The graded payload is per **chunk**, so two
scripts now follow one pair — (query, relevant session) — through each stage and
report where it dies: `scripts/attribute_recall_loss.py` (session unit) and
`scripts/diagnose_entry_level.py` (chunk unit, which defines a *gold entry* as a
memory whose extracted identifier is a file the task's patch touched).

Session unit, shipped configuration (cap 5, `evidence_max_sessions` 2, dense +
rerank on GPU, 333 relevant pairs over 89 queries; strong = ≥2 shared files):

| stage | ALL survive | STRONG survive | lost at this stage |
|---|---|---|---|
| ≥1 entry in the recall pool | 94.3% | 99.0% | 5.7% |
| ≥1 entry admitted for scoring (lexical/entity hit) | 85.3% | 94.1% | 9.0% |
| ≥1 entry in the payload with the session budget lifted | 80.5% | 93.1% | 4.8% |
| ≥1 entry emitted under `evidence_max_sessions=2` | **27.6%** | 49.5% | **52.9%** |
| ≥1 entry inside the 10-slot window | 27.6% | 49.5% | 0.0% |

The replay is exact: computing recall from these stages reproduces the live
benchmark to the digit (0.475 vs the measured 0.4746 at 2 sessions; 0.662 vs
0.6624 at 6). Three things it settles:

- **Candidate generation is not the binding constraint, and neither is the
  window.** 94.3% of relevant sessions are already pooled, and the last row's
  zero is structural — the payload now averages 9.2 entries, so it *is* inside a
  10-slot window. The old "reordering owns 59 % of the shortfall" decomposition was
  measured at cap 3 with sessions uncapped (59 entries over 28.6 sessions), where
  the window and the payload were different objects. It no longer applies.
- **Emitting two sessions is what costs the recall**, and it is a policy choice:
  lifting the budget to 6 moved the proxy from 0.4746 to 0.6624, while precision@10
  fell 0.52 → 0.31 and the payload grew 9.2 → 24.4 entries.
- **Ranking headroom is separable from breadth.** With a *perfect* session ranking,
  2 sessions would reach 0.694 (against 0.475 achieved), 6 would reach 0.882, and
  unlimited breadth tops out at 0.900 — so ~22 points are available from ranking at
  any budget, and the remaining 10% of pairs are genuinely never pooled.
  16 of 89 queries currently ship a payload containing **no** relevant session, and
  all 16 are ranking failures: a gate-cleared relevant session existed and was
  ranked third or worse.

Chunk unit is much worse, and this is the number the session view hides. Of the 333
relevant sessions: 85.9% contain at least one gold entry in the corpus, but only
**40.5% have one in the recall pool**, and only **13.5% have one emitted** (strong
labels: 95.0% → 65.3% → 27.7%). A relevant session averages 102 entries of which
3.8 name a task file; 17.4 of its entries get pooled and 0.9 of those are gold —
i.e. in 43.6% of the pooled cases the system grabbed a dozen siblings and missed the
one chunk that matters. And the chunk that *carries the session's score* (its
highest-ranked entry, which is what the session ranking compares) is a gold entry
only **12.9%** of the time.

**Deepening recall is zero-sum on this axis — measured.** Pulling
`recall_per_channel` 120 → 600 and `candidate_pool` 300 → 2400 lifts gold-in-pool
from 40.5% to 62.5%, but the deeper pool is mostly further siblings of the sessions
already there, so gold reaching the payload falls 68.9% → 39.9% and the end-to-end
figure is unchanged (13.5% → 13.2%). This corrects the earlier reading of the
pool 300 → 800 experiment ("the cause is weak textual evidence, not truncation"):
that was decided on `recall@100`, which is already saturated in the session view and
structurally cannot see chunk identity. The conclusion stands, the evidence for it
had to be re-derived.

**Why re-weighting cannot fix the 12.9%.** Decomposing the leading session's head
against a *losing* strong-relevant session's best gold entry (66 pairs):

| term | head | gold entry | gap |
|---|---|---|---|
| normalized fused rank (`base`) | 0.872 | 0.623 | +0.249 |
| channel `strength` | 0.844 | 0.621 | +0.223 |
| `coverage` | 0.644 | 0.583 | +0.061 |
| identifier evidence (`entity_signal`) | 0.224 | 0.117 | +0.107 |
| final, pre-rerank | 0.826 | 0.606 | +0.221 |
| final, post-rerank | 1.000 | 0.676 | **+0.324** |

The gold entry loses on **all four** terms, and only 12.1% of them beat the leading
head even before the cross-encoder ran. So no non-negative re-weighting of the
existing blend can order these two correctly: this is a feature-space limit, not a
calibration problem. (The cross-encoder, incidentally, *widens* the gap here —
+0.221 → +0.324 — consistent with the session-level reranking result above: it is
good at choosing among a session's chunks and bad at comparing sessions, because it
sees one chunk at a time.)

### Two attempts to move the 12.9%: one rejected, one shipped

Both target the fact that a session's score is carried by whichever of its ~100
chunks matches the question loudest, which is usually a read of a file and not the
edit that changed it.

**`operative_rank_weight` — a fifth scoring term, measured worse, off by default.**
The corpus says the decisive rows are rare and specific: among messages naming a
file the task touched, only 17.8% record an edit, and `code` is 81% of the corpus.
So a term for "this chunk records an action" was tried as a scoring term:

| weight | head is a gold entry | gold emitted | head−gold score gap |
|---|---|---|---|
| 0 | 12.9% | 33.3% | +0.221 |
| 0.1 | 12.3% | 33.3% | +0.237 |
| 0.2 | 11.7% | 32.6% | +0.272 |
| 0.3 | 10.8% | 31.1% | +0.306 |

Monotonically the wrong way, for a reason the same dump shows: **the chunk that beat
it is also an action record** — mean operative score 0.04 apart — so an unconditioned
action term lifts the distractors as much as the answer. This is the same failure as
promotion applied to every session (ambiguity 0.367 → 0.567). Rewritten as the
interaction "records an action **and** names an identifier the question names" it
went flat (head is a gold entry 12.9% → 12.9/13.2/12.9/12.6 across 0.1–0.5). Both
forms stay switchable and off; the knob is retained because it is the cheapest way to
re-test this if the corpus changes.

**`evidence_position_weight` — intra-session, shipped at 1.0.** The signal that does
separate reads from edits is *where* in the trajectory a chunk sits. Corpus
measurement (300 sessions, no retrieval involved): the share of messages naming a
patched file rises from 3.0% in the first decile of a session to 21.9% in the
seventh; mean relative position 0.575 for such messages, 0.661 for the edits among
them, 0.500 for all messages. Trajectories read before they change.

The tilt applies only inside a session already chosen — `final × (1 + w·(rel−0.5))`
over the chunk's row-id span within its session (`Store.session_span`) — so it cannot
buy a distractor session a slot, which is exactly what killed the scoring-term form.
Denominator: the 135 relevant sessions that have a gold entry in the pool.

| weight | gold emitted | gained / lost | exact McNemar p | `decidable` (n=30) | `ambiguous` |
|---|---|---|---|---|---|
| 0 (was shipped) | 33.3% | — | — | 0.500 | 0.333 |
| 0.6 | 39.3% | +8 / −0 | 0.0078 | 0.500 | 0.300 |
| **1.0 (shipped)** | **42.2%** | **+12 / −0** | **0.0005** | **0.567** | **0.267** |
| 1.5 | 44.4% | +15 / −0 | 0.0001 | 0.600 | 0.267 |
| 2.0 | 44.4% | +15 / −0 | 0.0001 | 0.600 | 0.267 |

Zero losses at every weight, and the strong-label subset agrees (28 → 37 at 1.0,
+9/−0, p=0.0039). Query-macro paired bootstrap +0.111 [+0.053, +0.179], p<0.001.
`decisive_present` rose 0.733 → 0.767 while `ambiguous` fell, so the later chunk is
not the noisier one. Payload size did not move at all (9.2 entries, ~1.86k tokens in
the evidence run).

**1.0, not the measured optimum 1.5.** The tilt spans `[1−w/2, 1+w/2]`; at 2.0 the
first chunk of a session has its score annihilated. 1.0 caps the first-to-last
difference at 3× and takes 80% of the plateau, and "zero losses" is 135 sessions of
one corpus, not the platform's distribution. 1.5 is recorded as the measured optimum
for whoever gets an end-to-end answer run to price it against.

**What this change cannot be credited with.** It moves no proxy metric: all eleven
benchmark numbers are identical to four decimal places, 0 of 89 queries changed, and
that is expected — relevance is labelled per session there, and this only decides
which chunk of a session is shown. `scripts/paired_compare.py` is what proves the
invariance (it also reproduces the recorded cap 3 → 2 `item_recall@10` delta of
+0.0498 exactly, so its zero is not a broken comparison).

**One side effect, caught by the suite.** The later chunk is sometimes a bare code
block, so item #1 can now be the code rather than the sentence that names the file.
`test_relevant_memory_ranks_first_despite_newer_distractors` failed for that reason
and was widened from matching the word "checkout" to matching any of the relevant
session's chunks: which chunk leads is the assembler's business, which *session*
leads is what that test exists to protect.

**Promotion is not redundant: the two levers act on different things.**
`evidence_operative_promotion` was introduced for cap 3, when the edit sat eighth in
its own session and never reached a slot, and on the *slot-membership* axis it no
longer earns its keep with cap 5 — promotion off alone moves gold-emitted 33.3% ->
34.1% (one session, noise), and with the tilt on, 43.0% without promotion against
42.2% with it. That measurement is narrower than it looked. `run_evidence.py` scores
what the answer model can read inside a token prefix, which is an *ordering* question
inside the session, and there promotion is decisive: at tilt 1.0, promotion 2 -> 0
takes `decisive_present` 0.733 -> 0.433 and `decidable` 0.600 -> 0.333 at a 1 000-token
prefix, and 0.567 -> 0.367 across the whole payload (30 questions, so roughly nine moved
the wrong way), while the payload shrinks 1 840 -> 1 637 tokens. So the tilt gets the right chunk *into* the payload
and promotion puts it *first*; the proxy's "gold emitted" axis can only see the
former, which is why it read as redundancy. Both stay on.

### Tuning decisions taken from measurements, not intuition

| Decision | Evidence |
|---|---|
| Keep session-major emission; reject entry-order assembly (2026-10-02) | Over a byte-identical item set (9.2 entries/query, 0 precision/recall movement), global score order does improve the ordering the platform cuts: item nDCG@10 +0.0190, item MRR +0.0257, both p<0.001 (`scripts/exp_entry_order.py`, S2 vs S0). But the shipped block order carries the position tilt and operative promotion, and on the evidence metric the entry walk without them loses decidable 0.567 → 0.300 (0 gained / 8 lost, p=0.0078; @1k prefix 0.633 → 0.167) with payload size unchanged. The proxy's gain is an order of magnitude smaller than the evidence loss; the walk order is not reopened unless the intra-session levers are ported into it and re-priced on `run_evidence.py`. The no-quota arm confirms the per-session cap is load-bearing (40.3 entries, 20.2/session flooding). |
| Rerank window resolves with the checkpoint (2026-09-27) | bge-reranker-v2-m3 at the MiniLM-era 512/200 window is a regression against the shipped MiniLM default (decidable 0.567 → 0.400, −8/+3): 76.7 % of Edit/Write/MultiEdit memories exceed 200 tokens, so the budget clips exactly the operative evidence. Opening the window to 2048/800 recovers and passes it (0.633, +9/−2, one-sided p=0.033; entry nDCG@10 p=0.074) at +0.7k payload tokens and a latency price that is hardware-shaped (234 s/search CPU vs ~3 s GPU). So `rerank_max_length` / `rerank_doc_tokens` / `rerank_probability_scores` default to None and resolve per checkpoint (bge-reranker-v2-m3 → 2048/800/on; else 512/200/off); explicit values always win. The proxy's recall metrics under-claim the effect because they are labelled per session — the same session-vs-entry blindness as the position tilt. See the equal-pool section. |
| Cap 5 per session, at most 2 sessions, multi-span top-1 (2026-09-25) | `decidable@all` 0.400 → 0.500, ambiguity 0.433 → 0.333, mean payload 7 190 → 2 434 tokens (−66 %) with `decidable@2k` unchanged — resolves the cap-3 stand-off by capping *sessions* instead of *slots per session*. Retrieval-proxy numbers elsewhere in this file predate the change; re-derive with `run_benchmark.py` before quoting them. |
| Cap items per session at 3 | Recall@100 rose 0.845 → 0.919 and nDCG@100 0.681 → 0.705. Without a cap, ~100 returned chunks collapsed to ~23 distinct sessions, starving other relevant work. (Historical; superseded by the 2026-09-25 row above.) |
| Reject the optimum at cap=1 | cap=1 measured marginally better recall (0.9231 vs 0.9193) but the proxy scores *whether a session was found*, not *whether its content is enough to answer*. Optimizing a measurable proxy at the cost of an unmeasurable quality is how benchmarks get gamed; cap=3 keeps session context for a 0.4 % metric difference. |
| Rerank weight 0.65, temperature 2.0 | Both swept. Weight: 0.65 peaks (MRR 0.7716); 0.85 and 0.95 degrade (0.7311, 0.7347) even though precision@10 rises — precision@10 is not what the answer model needs. |
| End-to-end result reported as null, not as a win | 0.689 vs 0.689 with 4 gained / 4 lost and p=0.72 is indistinguishable from chance. Reporting the aggregate as anything other than "no detectable effect" would be reading noise, and the paired no-memory condition exists precisely to make that visible. |
| Session-major candidate pool off by default | Measured on the same machine: MRR +0.019 (p=0.25) against item recall@10 −0.019 (p=0.25) — noise in both directions, and the one thing it clearly does (widen the payload) is the thing the earlier pool 300→800 experiment showed is not the binding constraint. |
| Session-level reranking off | Every metric sits below the entry-level stage (MRR −0.059, p=0.006) and lowering the blend weight drifts monotonically back toward the baseline — the attenuation signature of noise, not signal. A single chunk does not stand for a 96-entry session. |
| `max_evidence_per_session` left at 3 | cap 2 measures item recall@10 +0.0498 (p<0.001), but with zero queries changing session MRR it is a window effect, not a ranking gain. Priced against `run_evidence.py`, every prefix row of cap 3 is ≥ cap 2 and `decisive` falls monotonically as the cap tightens (0.733 → 0.700 → 0.633 → 0.500 at cap 1). The gain is on a proxy of window coverage, the loss on a proxy of usefulness — so the default stays. **Superseded 2026-09-25: the cap moved to 5, with the window-coverage gain instead obtained by capping distinct sessions at 2; see the multi-span section.** |
| Listwise reranking off by default | Every setting scored below the no-listwise configuration and cost ~5x search latency. See the table above; the attenuation signature shows it adds noise here, but the proxy measures file overlap rather than usefulness, so this is unresolved rather than settled. |
| Dense enabled by default, device `auto` | The dense cost/benefit flips with hardware (see the table above). `auto` resolves to CUDA when present and CPU otherwise, so one image is fast on a GPU host and still contract-compliant on a CPU one, rather than being tuned for whichever machine happened to measure first. |
| Rerank pool 120 | MRR 0.684 / 0.746 / 0.772 at top_n 30 / 60 / 120: larger is better, and on GPU the 120-pool costs 0.5 s per search, so there is no reason to shrink it. |
| Fixed rerank temperature, not max-normalisation | Normalising by the head's maximum score made every contribution depend on which items happened to be reranked, so changing `rerank_top_n` produced an incoherent sequence (MRR 0.818 → 0.772 → 0.684 as the pool grew). A fixed temperature makes the mapping absolute; the sequence is now monotone (0.684 → 0.746 → 0.772 for top_n 30 → 60 → 120). This also means the earlier 0.8183 figure was an artifact of the flawed normalisation, which is why every number above was re-measured. |
| Clip rerank documents by token count, in one batch | Characters are a bad cost proxy on this model family: 2 048 characters can be 630 tokens while 40 characters is 21, and latency scales with real tokens (4 ms/doc at 21 tokens, 48 ms/doc at 1 034). Token clipping took long-memory reranking from 48 to ~14 ms/doc. Batching the tokenizer call (120 docs in one call rather than 120 calls) took the pool of 120 from 6.0 s to 2.1 s. |
| Keep the noise gate at 0.15 | On this benchmark gate=0 and gate=0.15 score identically, because lexical/entity recall already bounds the candidate set: the gate is not the active constraint here. It is retained because its purpose is the *unrelated-query* case, which this dataset does not exercise — that case is covered by `tests/test_ranking_scale.py` with synthetic same-repo noise. |
| Promote the operative chunk, but only within the top session | Promoting it in every session raised decisive evidence 0.500 → 0.667 and ambiguity 0.367 → 0.567 simultaneously, netting *below* baseline. Restricted to the session we already rank first it gained on all three axes (0.567 / 0.333 / 0.333). The same lever applied everywhere is the same lever applied to evidence we do not believe. **Extended 2026-09-25:** with at most 2 sessions admitted, promotion now covers the top 2 (`evidence_operative_promotion=2`). |
| Do not cap distinct sessions (`evidence_max_sessions=0`) | The prediction was that tightening the cap buys precision with coverage we could spare, since the answer session was retrieved 100% of the time. Wrong: it is in the payload 100% of the time but is the *top-ranked* session only 70% of the time, so `decidable` stayed flat while decisive fell. What actually needs work is session ranking, not assembly breadth. **Under the prefix metric this became much less clear** — `max_sessions=1` scores the same `decidable` at 676 tokens instead of 6 518. **Superseded 2026-09-25: the cap ships at 2** (`decidable@all` 0.400 → 0.500, ambiguity −23 %, payload −66 %); an end-to-end Answer re-test is still owed. |
| Session-major assembly is free on the retrieval proxy | MRR 0.7800 and recall@10 0.7186 unchanged, nDCG@100 0.7306 → 0.7368, items 67 → 59. File-overlap ground truth cannot distinguish which chunk of a session leads, so a change that only affects chunk *identity* inside a session shows up as no cost. |
| Chunk kind labels tightened against the corpus audit | 95% of `config` chunks were the task prompt and 43% of `diff` chunks had no hunk at all (`scripts/audit_chunk_kinds.py`). Kept on index-correctness grounds: the flat-to-mixed metric movement (recall@10 +0.0028, nDCG@10 −0.0075) is within one query, and the label decides which entity extractors run, which file overlap cannot price. |
| No `code` bonus for debug intent | Operative evidence is concentrated in `code` chunks (2 869 of 3 464), but weighting the kind cost recall@10 (0.7214 → 0.7115), precision@10 (0.2323 → 0.2267) and mean prefix `decidable` (0.408 → 0.383) while only MRR rose. A kind label cannot isolate a 9% minority inside it. |
| Intra-session position tilt on at 1.0 (`evidence_position_weight`) | The chunk carrying a session's score names a task file only 12.9% of the time; tilting slot choice toward the end of that trajectory takes it to 42.2% (+12 gained / −0 lost, exact McNemar p=0.0005; strong labels +9/−0), raises `decidable` 0.500 → 0.567 and lowers `ambiguous` 0.333 → 0.267, with payload size unmoved. 1.5 measures better (44.4%) but at w=2.0 the first chunk's score is annihilated, so the bound decided. The proxy benchmark cannot see this change at all — which is expected, not a null result. |
| Operative evidence as a fifth **scoring** term, off | It made its own target worse, monotonically in the weight (head-is-gold 12.9% → 10.8% at 0.3), because the chunk that beat the gold entry is itself an action record (operative scores 0.04 apart). Rewritten as "action **and** a queried identifier": flat across 0.1–0.5. Kept switchable, not shipped. |
| Widening recall depth/pool is not the entry-level fix | `recall_per_channel` 120 → 600 with pool 300 → 2400 lifts gold-in-pool 40.5% → 62.5% and costs gold-in-payload 68.9% → 39.9% — end to end 13.5% → 13.2%, i.e. zero, because the deeper slice is more siblings of sessions already present. The older pool 300 → 800 conclusion was right but had been read off a session metric that structurally cannot see chunk identity. |
| Session-presence rates are not delivery rates | "94.3% of relevant sessions are pooled" means ≥1 chunk of ~102. Every funnel claim now has to be read at chunk level (`scripts/diagnose_entry_level.py`), where the same stage is 40.5%. |
| `evidence_operative_promotion` stays at 2, though it measures redundant on one axis | With the position tilt on, turning promotion off *helped* the slot-membership proxy (gold emitted 42.2% -> 43.0%) and wrecked the prefix metric: `decisive_present` 0.733 -> 0.433 and `decidable` 0.600 -> 0.333 at a 1 000-token prefix (0.567 -> 0.367 over the whole payload; n=30, about nine questions the wrong way), payload 1 840 -> 1 637 tokens. The tilt decides which chunk gets into the payload; promotion decides which one the answer model reads first. A proxy that can see only the former is not a licence to delete the latter. |

### What this benchmark cannot tell us

- **It is retrieval-only.** There is no Answer/Eval stage here, so it cannot show
  whether a returned memory helps the answer model. That requires the P3
  end-to-end harness.
- **The relevance definition may not match the organisers'.** If CAMBench credits
  a session for a shared *technique* rather than shared *files*, our lower-bound
  ground truth understates performance.
- **It is not the scored suite.** Never present these numbers as official.
