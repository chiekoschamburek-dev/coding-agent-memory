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

Measured on the development GPU (RTX 5060 Laptop, 8 GB) with `device=auto`:

| configuration | MRR | nDCG@10 | nDCG@100 | recall@10 | precision@10 | Add | Search |
|---|---|---|---|---|---|---|---|
| random baseline | 0.2334 | 0.1936 | 0.3804 | 0.2919 | 0.0972 | — | — |
| **P1** lexical + entity | 0.7248 | 0.6138 | 0.7047 | 0.6655 | 0.1972 | 56 s | 47 ms |
| P2a + dense | 0.7659 | 0.6421 | 0.7252 | 0.6885 | 0.2073 | 163 s | 224 ms |
| P2b + rerank | 0.7716 | 0.6474 | 0.7296 | 0.6914 | 0.2135 | 41 s | 521 ms |
| **P2c + dense + rerank** (default) | **0.7718** | **0.6586** | **0.7306** | **0.7133** | **0.2242** | 155 s | 674 ms |

MRR 0.77 against 0.23 for random, nDCG@10 0.66 against 0.19. Since the platform
consumes a token-counted *prefix* of our ranking, the `@10` figures are closer to
what reaches the answer model than `@100` is.

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

### End-to-end Answer evaluation

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

### Tuning decisions taken from measurements, not intuition

| Decision | Evidence |
|---|---|
| Cap items per session at 3 | Recall@100 rose 0.845 → 0.919 and nDCG@100 0.681 → 0.705. Without a cap, ~100 returned chunks collapsed to ~23 distinct sessions, starving other relevant work. |
| Reject the optimum at cap=1 | cap=1 measured marginally better recall (0.9231 vs 0.9193) but the proxy scores *whether a session was found*, not *whether its content is enough to answer*. Optimizing a measurable proxy at the cost of an unmeasurable quality is how benchmarks get gamed; cap=3 keeps session context for a 0.4 % metric difference. |
| Rerank weight 0.65, temperature 2.0 | Both swept. Weight: 0.65 peaks (MRR 0.7716); 0.85 and 0.95 degrade (0.7311, 0.7347) even though precision@10 rises — precision@10 is not what the answer model needs. |
| End-to-end result reported as null, not as a win | 0.689 vs 0.689 with 4 gained / 4 lost and p=0.72 is indistinguishable from chance. Reporting the aggregate as anything other than "no detectable effect" would be reading noise, and the paired no-memory condition exists precisely to make that visible. |
| Listwise reranking off by default | Every setting scored below the no-listwise configuration and cost ~5x search latency. See the table above; the attenuation signature shows it adds noise here, but the proxy measures file overlap rather than usefulness, so this is unresolved rather than settled. |
| Dense enabled by default, device `auto` | The dense cost/benefit flips with hardware (see the table above). `auto` resolves to CUDA when present and CPU otherwise, so one image is fast on a GPU host and still contract-compliant on a CPU one, rather than being tuned for whichever machine happened to measure first. |
| Rerank pool 120 | MRR 0.684 / 0.746 / 0.772 at top_n 30 / 60 / 120: larger is better, and on GPU the 120-pool costs 0.5 s per search, so there is no reason to shrink it. |
| Fixed rerank temperature, not max-normalisation | Normalising by the head's maximum score made every contribution depend on which items happened to be reranked, so changing `rerank_top_n` produced an incoherent sequence (MRR 0.818 → 0.772 → 0.684 as the pool grew). A fixed temperature makes the mapping absolute; the sequence is now monotone (0.684 → 0.746 → 0.772 for top_n 30 → 60 → 120). This also means the earlier 0.8183 figure was an artifact of the flawed normalisation, which is why every number above was re-measured. |
| Clip rerank documents by token count, in one batch | Characters are a bad cost proxy on this model family: 2 048 characters can be 630 tokens while 40 characters is 21, and latency scales with real tokens (4 ms/doc at 21 tokens, 48 ms/doc at 1 034). Token clipping took long-memory reranking from 48 to ~14 ms/doc. Batching the tokenizer call (120 docs in one call rather than 120 calls) took the pool of 120 from 6.0 s to 2.1 s. |
| Keep the noise gate at 0.15 | On this benchmark gate=0 and gate=0.15 score identically, because lexical/entity recall already bounds the candidate set: the gate is not the active constraint here. It is retained because its purpose is the *unrelated-query* case, which this dataset does not exercise — that case is covered by `tests/test_ranking_scale.py` with synthetic same-repo noise. |

### What this benchmark cannot tell us

- **It is retrieval-only.** There is no Answer/Eval stage here, so it cannot show
  whether a returned memory helps the answer model. That requires the P3
  end-to-end harness.
- **The relevance definition may not match the organisers'.** If CAMBench credits
  a session for a shared *technique* rather than shared *files*, our lower-bound
  ground truth understates performance.
- **It is not the scored suite.** Never present these numbers as official.
