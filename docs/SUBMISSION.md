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
| `session_feature_fusion` | `false` | significant on the proxy (recall +0.061, p=0.001) and the largest reach gain (+12.1 pp), but flat on answers and did not convert as select's shortlist feeder — documented lever, not shipped |
| `session_select_llm` / `session_select_timeout_seconds` | **`true` / 10.0** | **the deployment default** — see §2 |

## 2. The settled ship decision: `session_select_llm` on

Seven session-ranking arms were measured on the 58-question procedure e2e
(n=58 × 5, majority vote; full detail in eval/README.md). Final leaderboard:

> RAG 0.224 < no-memory prior 0.328 < feature-fusion 0.345 < shipped 0.362 <
> gate-0.45 = fusion+select 0.397 < **select-llm 0.414**

**select-llm is the only arm with triangular support** — significant on the
proxy (session MRR 0.7584 → 0.8652, p<0.001; recall +0.043, p=0.035),
direction-positive on answers (+4/−1, the largest single-arm gain), and
mechanically aligned with the graded metric (it makes the Answer model's own
judgement — "can this context answer this question" — over session-level
summaries, one gpt-4o-mini call per Search, the model the rules fix for the
graded path). By the same two-sided p<0.05 standard that kept
`dense_eligible` out of the defaults, it is the deployment default: it
degrades to the shipped ordering on any relay absence, timeout (hard 10 s
budget), or unparseable reply, so shipping it enabled is fail-safe.

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
- `dense_eligible@0.45` remains the **smoke-window A/B arm** (directional
  +2/−0, one verified mechanism case, one env flip with measured rollback).
  Never combine with `dense_eligible_min_similarity=0.30` (reversal measured).

## 3. Evidence inventory (what we can defend, at what strength)

| claim | strength | source |
|---|---|---|
| The gated pipeline beats pure RAG on memory-dependent questions, **p=0.0063** | significant | 58-question procedure e2e: RAG 0.224 (below chance) vs gated 0.397, same dense channel, 1/17 the payload |
| "Return everything" is actively harmful where memory matters | significant, replicated | RAG below the no-memory prior on both set sizes (32 q and 58 q) |
| `dense_eligible@0.45` improves answer accuracy | directional, 3 wins / 0 losses across two instruments, mechanism verified | §2 |
| Rerank window travels with the checkpoint | paired, one-sided p=0.033 | equal-pool section |
| Position tilt lifts the operative chunk | paired, p=0.0005, zero losses | position section |
| Session ranking, not recall, is the binding constraint | measured funnel: 94.3 % pooled, ideal ordering 0.694 vs 0.475 actual | attribution section |
| Cards, listwise reranking, entry-order emission, operative scoring term | measured negative, documented and switchable | respective sections |

Instrument limits, recorded so results are never over-read: the file-
localisation e2e is prior-saturated (no-memory 0.700) and cannot show
memory value; the procedure set caps at 58 questions (the corpus's
recorded-cause prose runs out — 14.7k of candidate messages are tool-call
records); its no-memory floor is 0.328, not 0.25 (option leak), which
compresses every arm gap. CAMBench is not public; nothing here is an
official number.

## 4. Deployment checklist (rules.md §6, the 8 items)

| # | item | status |
|---|---|---|
| 1 | Smoke test passed | **open** — needs the deployed endpoint; every track self-hosts since 2026-09-26. During the soak: quantify the select-llm fallback rate (the `session selection failed` warning) and the answer-call error rate under real network |
| 2 | API contract correct | covered: `tests/test_contract.py` (identifier echo, durability-then-success, top_k ceiling, error envelope, auth, isolation, idempotency, concurrency) |
| 3 | Add/Search models = `gpt-4o-mini` | the graded path uses no LLM by default; the optional LLM (query probes, enrichment, listwise) defaults to `gpt-4o-mini` (`CODEMEM_LLM_MODEL`); the dense/rerank encoders are non-generative, disclosed in `docs/COMPLIANCE.md` |
| 4 | ≥30 days publicly reachable | **open — the long pole**: VM direct port exposure per `deploy/README.md` (CDN/edge proxies cap timeouts below the worst-case Add and are recorded as unsafe) |
| 5 | Run instructions complete | `README.md` (quick start, API, config), `deploy/README.md` (image, ingress), `.env.example` (every flag) |
| 6 | Originality disclosed | method changes are itemised in `eval/README.md` (each with its measurement); the paper survey supporting the design is part of the submission materials |
| 7 | Substantive | non-trivial measured method with negative results documented |
| 8 | No manipulation | Search never generates returned text (`tests/test_traceability.py`); `user_id` isolation on every read/write path (`tests/test_isolation.py`); the proxy benchmark is labelled not-the-scored-suite in its own metadata |

## 5. Parked (with reasons)

- **Issue-language card overview** (HyDE-style: describe the problem the
  session solves, in issue vocabulary): sharpened by the miss decomposition
  — the 31 select-llm misses split into 20 shortlist-bottleneck + 11 summary
  quality, and the card attacks both (a new candidate kind in the pool; an
  issue-language summary replacing the digest). **Carry the stack's warning**:
  the LLM judge is non-monotone in shortlist composition, so a card
  evaluation must measure reach and answers directly, never intermediate
  counts ("more relevant sessions in the shortlist" misled once already).
  Restart only after the deployment window is locked and time allows.
- **Budget × gate interaction on the procedure e2e** (`evidence_max_sessions`
  3–4 with `dense_eligible@0.45`): never swept under the answer-accuracy
  lens; the 32-question runs cannot resolve it.
- **Instrument growth**: the 58-question ceiling binds every future
  comparison; a second anchor strategy (claim-topical rather than
  file-overlap relevance) is the path past it, at the cost of weakening the
  ground truth.
