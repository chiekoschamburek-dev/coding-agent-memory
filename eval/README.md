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

300 sessions added (42.9 s), 89 scored queries, 53.7 ms mean search latency.

| metric | random baseline | codemem | lift |
|---|---|---|---|
| MRR | 0.2334 | **0.7248** | +0.4914 |
| nDCG@10 | 0.1936 | **0.6138** | +0.4202 |
| nDCG@100 | 0.3804 | **0.7047** | +0.3243 |
| recall@10 | 0.2919 | **0.6655** | +0.3736 |
| recall@100 | 0.9484 | 0.9193 | −0.0291 |
| precision@10 | 0.0972 | **0.1972** | +0.1000 |

Ranking quality is the headline: MRR 0.72 against 0.23 for random, and nDCG@10
0.61 against 0.19. Since the platform consumes a token-counted *prefix* of our
ranking, `@10` figures are closer to what actually reaches the answer model than
`@100` is.

**Why recall@100 is below random.** The random baseline returns its entire
shuffled pool averaged over many random orders, so it eventually includes almost
every session and recall@100 saturates near 1.0. That is an artifact of the
metric rewarding volume, not retrieval skill. Two observations make the
comparison hollow:

- at the rank positions that matter (`@10`) we are 2.3× random;
- of 89 queries, the median rank of the first relevant memory is **1**, the 90th
  percentile is 7, and only 2 queries fail to surface any relevant memory at all.

Because the random baseline's pool is the full session list, its per-query
variance is enormous — which is also why the harness now refuses to print metrics
if the random baseline scores exactly zero, a state that can only mean the
comparison itself is broken.

### Tuning decisions taken from measurements, not intuition

| Decision | Evidence |
|---|---|
| Cap items per session at 3 | Recall@100 rose 0.845 → 0.919 and nDCG@100 0.681 → 0.705. Without a cap, ~100 returned chunks collapsed to ~23 distinct sessions, starving other relevant work. |
| Reject the optimum at cap=1 | cap=1 measured marginally better recall (0.9231 vs 0.9193) but the proxy scores *whether a session was found*, not *whether its content is enough to answer*. Optimizing a measurable proxy at the cost of an unmeasurable quality is how benchmarks get gamed; cap=3 keeps session context for a 0.4 % metric difference. |
| Keep the noise gate at 0.15 | On this benchmark gate=0 and gate=0.15 score identically, because lexical/entity recall already bounds the candidate set: the gate is not the active constraint here. It is retained because its purpose is the *unrelated-query* case, which this dataset does not exercise — that case is covered by `tests/test_ranking_scale.py` with synthetic same-repo noise. |

### What this benchmark cannot tell us

- **It is retrieval-only.** There is no Answer/Eval stage here, so it cannot show
  whether a returned memory helps the answer model. That requires the P3
  end-to-end harness.
- **The relevance definition may not match the organisers'.** If CAMBench credits
  a session for a shared *technique* rather than shared *files*, our lower-bound
  ground truth understates performance.
- **It is not the scored suite.** Never present these numbers as official.
