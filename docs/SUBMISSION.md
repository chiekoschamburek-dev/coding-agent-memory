# Submission decision record

Final measured configuration, evidence inventory, and the deployment
checklist for the Agent Memory Challenge (code track). Every claim here
points at the eval section that measured it; nothing in this document is
intuition. Companion documents: `docs/DESIGN.md` (architecture and
invariants), `docs/COMPLIANCE.md` (rules mapping), `deploy/README.md`
(self-hosting), `eval/README.md` (all measurements).

## 1. Shipped configuration (the defaults in `src/codemem/core/config.py`)

| setting | value | verdict it rests on |
|---|---|---|
| `dense_enabled` | `true` | largest single recall gain; device `auto` |
| `rerank_enabled`, model | `true`, MiniLM-L-6-v2 | largest measured gain (MRR +0.093) |
| `rerank_max_length` / `rerank_doc_tokens` / `rerank_probability_scores` | resolve with the checkpoint | equal-pool experiment: bge-v2-m3 at the MiniLM window was a regression (decidable 0.567 → 0.400); 2048/800 recovers it (0.633) |
| `rerank_top_n` / `rerank_weight` | 120 / 0.65 | swept |
| `evidence_position_weight` | 1.0 | +12/−0 gold emitted, McNemar p=0.0005 |
| `evidence_operative_promotion` | 2 | prefix metric decisive; the proxy cannot see it |
| `evidence_operative_weight` | 1.0 | span-selection weight for returned content (distinct from the retired scoring term below) |
| `max_evidence_per_session` / `evidence_max_sessions` | 5 / 2 | decidable 0.400 → 0.500, payload −66 % |
| `min_evidence_score` | 0.15 | unrelated-query guard, covered by tests |
| `listwise_enabled` | `false` | every setting below the no-listwise baseline, attenuation signature |
| `operative_rank_weight` | 0.0 | fifth scoring term, monotonically wrong direction |
| `card_enabled` / `card_expansion` | `false` / `false` | measured inert on all three harnesses at 9× Add cost; the "qualifies nothing" invariant blocks the recall path by construction |
| `dense_eligible` | `false` | directional (+2/−0 on the procedure e2e) but never significant; kept as the smoke-window A/B arm |
| `session_score_topk` | `1` (off) | top-k mass aggregator: proxy recall +0.019 (p=0.004), answers +1 q; the gate+top-2 combo was the best proxy ranking on record yet did not stack on answers — recorded best-not-shipped |
| `session_feature_fusion` | `false` | significant on the proxy (recall +0.061, p=0.001) and the largest reach gain (+12.1 pp), but flat on answers and did not convert as select's shortlist feeder — documented lever, not shipped |
| `session_select_llm` / `session_select_timeout_seconds` / `session_select_votes` | **`false`** / `10.0` / `1` | **reversed 2026-10-05, see §2** — seven-arm champion on the file-overlap instrument (+0.052) yet below the no-memory floor on the leak-free claim-topical one (−0.071); one env flip re-enables. Self-consistency (votes) measured dead: majority fixes variance, not bias |

## 2. The ship decision: `session_select_llm` — decided on, then reversed by its own generalisation test

Decided **on** 2026-10-04 on single-instrument triangulation; reversed
**off** 2026-10-05 when the second anchor (built precisely to cure the
selection-bias caveat recorded below) came back against it. Both halves
are kept — the reversal is the instrument-growth item doing its job.

**Why it shipped on (2026-10-04).** Seven arms on the 58-question
procedure e2e (n=58 × 5, majority vote):

> RAG 0.224 < no-memory prior 0.328 < feature-fusion 0.345 < shipped 0.362 <
> gate-0.45 = fusion+select 0.397 < **select-llm 0.414**

The only arm with triangular support: significant on the proxy (session
MRR 0.7584 → 0.8652, p<0.001; recall +0.043, p=0.035), direction-positive
on answers (+4/−1, the largest single-arm gain), mechanically aligned with
the graded metric (it makes the Answer model's own judgement over
session-level summaries, one gpt-4o-mini call per Search, fail-safe on
every relay failure mode). +4/−1 at n=58 is p=0.375 — the decision rested
on triangulation plus fail-safety, never on single-run significance, and a
pre-registered confirmation run reproduced 0.414 to the question.

**Why it reversed (2026-10-05).** The claim-topical instrument — a second
anchor built on recorded-claim embedding affinity instead of file overlap,
leak-free (no-memory floor 0.243 = chance, where the procedure set's
floor was 0.328) — flipped the ordering:

| arm | file-overlap anchor (58 q) | claim-topical anchor (70 q) |
|---|---|---|
| no-memory floor | 0.328 (option leak) | **0.243** (= chance) |
| shipped ordering (`max`) | 0.362 | **0.300** |
| `session_select_llm` | **0.414** | 0.229 — **below floor** |
| select + votes=3 | not run | 0.214 (worse still) |

Mechanism, confirmed by the one-call probe: the selection judgement
inherits the answer model's prior. The procedure set's leak pointed that
prior at the gold — alignment won. The claim set's adversarial
construction points it at the distractors (27 of 140 picks landed on the
question's own distractor sessions) — alignment amplified the error below
the no-memory line. Self-consistency died the same day with the cleanest
mechanism of the campaign: majority voting fixes variance, not bias, and
votes=3 entrenched the systematically-wrong picks.

**The shipped default is therefore `false`:** two instruments bracket the
platform's unknown distribution; at the leak-prone end select wins (+5.2),
at the leak-free adversarial end it loses (−7.1) and can fall below
no-memory — the cardinal sin under the RAG doctrine — while the
deterministic ordering is never below the floor on either. The env flip
(`CODEMEM_SESSION_SELECT_LLM=1`) and the fail-safe fallback path are
unchanged; if deployment smoke or platform feedback indicates a leak-prone
question style, the flip is one line with a measured number behind it.

Mechanistic negative results carried alongside (worth more than plain nulls):

- **fusion** is the proxy's second significant lever (recall +0.061, p=0.001)
  and the largest reach gain (+12.1 pp) — yet flat on answers, and feeding
  its ordering into select's shortlist **reduced** reach (46.6 → 43.1 %):
  the LLM judge is non-monotone in shortlist composition. Recorded so nobody
  re-derives "more relevant sessions in the shortlist ⇒ better answers".
- **the stack's death is informative**: its 6-question ceiling came from a
  replay of shortlist supply, and the measured outcome (−1 vs select alone)
  bounded what "known expected value" can promise — supply is necessary, not
  sufficient; the judge eats re-ranking deltas.
- **the other two stacks died cheap**: gate + top-2 was the best proxy
  ranking on record (recall 0.5013, p=0.012) yet did not stack on answers
  (combo 0.379 = top-2 alone; gate alone 0.397); gate + select was killed
  at zero cost by the miss decomposition — 0 of 31 misses were gate-blocked.
- `dense_eligible@0.45` remains the **smoke-window A/B arm** (directional
  +2/−0, one verified mechanism case, one env flip with measured rollback).
  Never combine with `dense_eligible_min_similarity=0.30` (reversal measured).

## 2a-bis. The noise law and its boundary (read before tuning anything)

The submitted configuration is deliberately compact, and the reason is
measured, not aesthetic: **on adversarial multiple-choice instruments,
retrieval reach buys matching confusion**. Four independent mechanism
families (feature fusion, LLM selection stacking, query-side HyDE, a
claims-only side-channel) each increased evidence reach and each failed to
improve — or reversed — answer accuracy on leak-free anchors; the
claims-channel experiment supplied the mechanism: when the options quote
corpus claims, the retrieved "relevant" claims include the distractor
options' own sources, so reach and confusion are the same material.

**This law is instrument-conditional.** It does not say retrieval reach is
bad in general. Its precondition — distractor options quoting corpus
claims — is a property of adversarial multiple-choice evaluation. On
non-adversarial, option-less queries the bill has never been observed, and
two retained flags are the ready levers for exactly that regime
(`CODEMEM_SESSION_FEATURE_FUSION`, `CODEMEM_CLAIM_CHANNEL`; offline data
supports both there). If the deployment or a future evaluation cycle sends
option-less queries, re-enable and re-measure before concluding anything.

## 2b. One-shot reality

There is **no platform-feedback loop this cycle**: one Full evaluation, no
iteration against its scores. Consequences, in order of weight:

- the frozen configuration above is **final for this cycle** — the
  algorithmic frontier map (eval/README.md: select-on reached 0.414 on the
  file-overlap anchor before the §2 reversal; 0.655 with perfect judgment;
  0.828 with perfect shortlists; 24 % of questions unreachable at any
  ordering) is next-cycle material, not post-deploy iteration;
- the confirmation run (0.414 reproduced to the question, zero relay
  failures) is the last accuracy evidence that can be gathered locally;
- the 30-day window is a **reliability** problem, not a tuning problem:
  load-tested with the selection stage on (deploy/README.md) — latency is
  360× inside the platform ceiling even when the relay queues, every
  fallback is invisible to clients, and the one silent failure mode found
  (CRLF in `.env` breaking every selection URL) is fixed in code. The soak
  metric is the `selection LLM call failed` warning count plus the relay
  balance, checked daily.

## 3. Evidence inventory (what we can defend, at what strength)

**Headline claim (revised 2026-10-05) — the conversion thesis and its
corollary:** aligning the selection stage with the answer judgement
converts reach into answers **when that judgement is right** — and
amplifies error when it is wrong. Evidence in both directions: pure RAG
holds decisive evidence in 93.3 % of payloads yet decidable collapses to
0.333 (noise, not starvation); fusion converts +12.1 pp of reach into −1
answer; gate + top-2 stacks on the proxy but not on answers; fusion→select
raised relevant-in-shortlist 1.03 → 1.24 while reach fell (the judge is
non-monotone in composition); select-llm converts on the leak-prone anchor
(+5.2) and falls below no-memory on the leak-free adversarial one (−7.1)
— the judgement inherits the answer model's prior, and a prior pointed at
distractors is amplified, not corrected.

| claim | strength | source |
|---|---|---|
| The gated pipeline beats pure RAG on memory-dependent questions, **p=0.0063** | significant | 58-question procedure e2e: RAG 0.224 (below chance) vs gated 0.397, same dense channel, 1/17 the payload |
| The gated ordering beats the no-memory floor on both anchors (+3.4 / +5.7 pp); select-llm's edge is prior-dependent (wins +5.2 leak-prone, loses −7.1 leak-free, below floor) | direction probe, n=70 with repeated issues | §2; claim-instrument sections |
| The 31 remaining select misses decompose 20 shortlist-bottleneck / 0 gate-blocked / 11 llm-judgment; of the 11, 8 flip on a single re-call (relay boundary jitter, not summary quality) | exact, zero-relay replay | decomposition + digest-replay sections |
| Selection self-consistency (votes) entrenches systematic bias: 0.214 < one-shot 0.229 on the claim anchor | measured, mechanism clean | claim-instrument section |
| "Return everything" is actively harmful where memory matters | significant, replicated | RAG below the no-memory prior on both set sizes (32 q and 58 q) |
| `dense_eligible@0.45` improves answer accuracy | directional, 3 wins / 0 losses across two instruments, mechanism verified | §2 |
| Rerank window travels with the checkpoint | paired, one-sided p=0.033 | equal-pool section |
| Position tilt lifts the operative chunk | paired, p=0.0005, zero losses | position section |
| Session ranking was the binding constraint on the file-overlap anchor (ideal 0.694 vs 0.475); on the claim anchor the **menu itself** binds (answer reaches top-8 for 22/70) | measured funnels on both anchors | attribution section; probe section |
| Cards, listwise reranking, entry-order emission, operative scoring term, both stacks, shortlist diversification, digest recomposition, votes | measured negative, documented and switchable | respective sections |

Instrument limits, recorded so results are never over-read: the file-
localisation e2e is prior-saturated (no-memory 0.700) and cannot show
memory value; the procedure set caps at 58 questions (the corpus's
recorded-cause prose runs out — 14.7k of candidate messages are tool-call
records); its no-memory floor is 0.328, not 0.25 (option leak), which
compresses every arm gap. The claim-topical second anchor (105 questions,
leak-free floor 0.243, 70 tune / 35 sealed) is a direction probe with
repeated issue texts — it decides sign, not significance. CAMBench is not
public; nothing here is an official number. **The 58-question instrument
is frozen (2026-10-04):** seven arms measured, every top gap ≤ 4 questions
and within noise — further cells would only inflate the
multiple-comparisons burden.

## 4. Deployment checklist (rules.md §6, the 8 items)

| # | item | status |
|---|---|---|
| 1 | Smoke test passed | **open** — needs the deployed endpoint; every track self-hosts since 2026-09-26. During the soak: the answer-call error rate under real network, and (if select-llm is enabled via env) the `session selection failed` warning rate |
| 2 | API contract correct | covered: `tests/test_contract.py` (identifier echo, durability-then-success, top_k ceiling, error envelope, auth, isolation, idempotency, concurrency) |
| 3 | Add/Search models = `gpt-4o-mini` | the shipped graded path uses no LLM (`session_select_llm` reverted to off — §2); when enabled via env it adds one selection call per Search (`CODEMEM_LLM_MODEL`, default `gpt-4o-mini`, `max_tokens=24` — choices only, never returned text); the dense/rerank encoders are non-generative, disclosed in `docs/COMPLIANCE.md` |
| 4 | ≥30 days publicly reachable | **open — the long pole**: VM direct port exposure per `deploy/README.md` (CDN/edge proxies cap timeouts below the worst-case Add and are recorded as unsafe) |
| 5 | Run instructions complete | `README.md` (quick start, API, config), `deploy/README.md` (image, ingress), `.env.example` (every flag, including `CODEMEM_SESSION_SELECT_LLM` and `CODEMEM_SESSION_SELECT_TIMEOUT_SECONDS`) |
| 6 | Originality disclosed | method changes are itemised in `eval/README.md` (each with its measurement); the paper survey supporting the design is part of the submission materials |
| 7 | Substantive | non-trivial measured method with negative results documented |
| 8 | No manipulation | Search never generates returned text (`tests/test_traceability.py`); the selection stage emits session choices only (`max_tokens=24`; guard tests `test_llm_select_reorders_sessions` and the fused-order stack guard in `tests/test_session_features.py`); `user_id` isolation on every read/write path (`tests/test_isolation.py`); the proxy benchmark is labelled not-the-scored-suite in its own metadata |

## 5. Parked (with reasons)

- **Issue-language card overview** (HyDE-style: describe the problem the
  session solves, in issue vocabulary): the justification **shifted from
  digest quality to recall**. The claim anchor's binding constraint is
  menu membership — the answer session reaches the top-8 for only 22/70,
  because the retrieval channels are file/vocabulary machinery and
  claim-topical anchors are often not file-overlap sessions — which is
  exactly what a session-level candidate object that embeds in query
  vocabulary attacks. The digest-quality half of the old case shrank to
  ~3 hard queries (8 of the 11 judgment misses were relay boundary
  jitter). **Carry the stack's warning**: the LLM judge is non-monotone in
  shortlist composition — measure reach and answers directly, never
  intermediate counts. Restart only after the deployment window is locked
  and time allows.
- **Budget × gate interaction on the procedure e2e** (`evidence_max_sessions`
  3–4 with `dense_eligible@0.45`): never swept under the answer-accuracy
  lens; the frozen 58-question instrument cannot resolve it without
  breaking the freeze.
- **Sealed holdout usage**: the claim instrument's 35 sealed questions are
  spent only on a final confirmation of whatever ships next cycle — one
  run, pre-registered, no tuning against them. (The instrument-growth
  item itself is no longer parked: the second anchor was built 2026-10-05
  and immediately earned its keep by reversing §2.)
