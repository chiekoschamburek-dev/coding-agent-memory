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

## 2a-ter. What 2026-10-07 closed, and the one figure that moved

Fourteen further arms were measured the same day (five bonus-table arms,
six session-budget arms, three answer-metric arms) and **none changed the
shipped configuration**. What they bought is the closure of four questions
and the correction of one number:

- **`INTENT_KIND_BONUS` stays.** The pairing shows `code` as the only
  per-entry feature where gold beats the entry that outranked it (+31.3 pp
  net) and `diff` sitting on the winner's side (−24.2 pp), which reads as a
  mis-pointed table. Replaying five tables moves 440-464 of ~1 000 candidate
  ranks and no outcome: `head_is_gold` spans 0.117-0.135, `gold_in_window`
  0.159-0.168, all p≥0.25. A lever that produces rank churn without outcomes
  is a wash, not an error. This also refutes "prefer action records" a third
  time, and the `dense_sim` row (−46.5 pp) forecloses the whole similarity
  family — the reason the cross-encoder widens the gold gap rather than
  closing it.
- **`evidence_max_sessions=2` stays, on a different argument.** Its own
  instrument now supports the *unlimited* arm's rejection (+1/−8 decidable,
  p=0.039; ambiguity +10/−0, **p=0.002**) but the shipped value is not the
  peak — `max_sessions=1` scores higher on `decidable` (0.633 vs 0.600). On
  the answer metric the two are a wash (+2/−3, p=1.000) because the setting
  is a **reach ↔ conversion dial**: narrowing to one session forfeits 20 pp
  of reach (0 gained / 14 lost, p=0.0001) to buy ~6 pp of conversion.
  The reason to keep 2 is that its maximum regret across every plausible
  answer-prefix length (0.034) is half either neighbour's (0.066) — under
  the platform's unpublished real prefix, 2 is the only arm never worse than
  0.600. The old justification ("companion move to cap 5") is superseded.
- **ms=3 was run on the answer metric and is not a reach play either.** The whole
  breadth axis is flat there — 0.314 / 0.329 / 0.314 at ms=1/2/3 — while reach
  climbs 22.9 → 42.9 → 57.1 % and acc-on-shown falls 0.688 → 0.533 → 0.425. The
  matched comparison explains it: the 14 questions whose decisive content lands
  in the first five items at ms=2 are the same 14 at ms=3 (14 ∩ 14; session one's
  order is untouched by seating a third session) and they lose 14.3 pp, while the
  13 served late at ms=2 gain 15.3 pp **without changing position**. Breadth
  redistributes correctness; it does not add any. The earlier reading of this axis
  as "reach is the untaken prize" is retracted — the marginal questions ms=3 newly
  serves convert at 0.100, below the 0.243 no-memory floor.
- **Abstention is closed as an algorithmic option.** Because accuracy on
  questions whose answer session is missing (0.175) sits *below* the
  no-memory floor (0.243), withholding would be worth +0.039 — but no
  absolute quantity the max-normalisation discards predicts it (IDF sum 0.403,
  BM25 0.563, dense 0.500, rerank logit 0.487, AUC floor ~0.64 at n=70).
  Resemblance is not presence: in a same-repository corpus some chunk always
  resembles the question, so score magnitude cannot tell whether the answer is
  among them.
- **`evidence_full_count` stays at 8.** Rendering the second session as
  pointers cuts payload tokens 28 % and improves the proxy (+1/−0) with an
  exactly identical answer outcome (0.329 → 0.329, reach and acc-on-shown
  unchanged). Side product: part of `decidable`'s ambiguity penalty is a
  string-parser artifact, not measured model confusion.
- **The pointer explanation for ms=3's null was wrong, and so is the recall gap
  this file leans on.** It was offered that breadth buys *nominal* reach because
  late items are clipped to 110-token pointers: measured against exact ground
  truth, **0 of 11** decisive items in the pointer zone hit the cap (median ~52
  tokens) — the entries are simply short. The `shown_full` column written to test
  that idea was deleted rather than left in place, and what the harness records
  instead is the answer session's position.
- **Admission-side recall work is closed by measurement, not parked.** With
  `gold_claim` (verbatim-verified 70/70) located at message level, the decisive
  sentence is in the recall pool and admitted for scoring for **70 of 70**
  claim-anchor questions, yet in the payload for 27. Of the 40 questions whose
  answer session is not shown, **0** have the content outside the pool. A new
  eligibility channel would have zero failures to attack. The 40.5 % figure that
  made recall look like the remaining surface is a file-overlap-anchor number, and
  that anchor's label prefers the distractor (winning session 1.49 vs losing
  relevant session 1.38).

**The figure that moved.** The per-session dedup fix (`7203a49`) shifted the
shipped evidence arm from the recorded **decidable 0.567 / ambiguous 0.300**
to **0.600 / 0.233**, with `decisive_present` and `session_retrieved`
unchanged. §3's numbers below that quote 0.567 predate it.

**A reading-frame change, not cosmetic.** Accuracy factors as
`P(shown) × acc|shown + P(absent) × acc|absent`, and the two factors move in
opposite directions under every presentation lever tried so far. Aggregate
accuracy therefore hides significant component moves (a p=0.0001 reach drop
presented as a p=1.000 accuracy verdict). Strata must be intersected before
comparing acc-on-shown across arms: the ms=1 shown set is a strict subset of
ms=2's, which is what made the conversion gain look like +15.5 pp when the
matched comparison is +6.3 pp. `pool_sessions` predicts correctness
independently of reach (p=0.011, holds in both strata) and is the covariate
to block future arms on.

## 2a-quater. Purity is the binding quantity (2026-10-08, read before proposing anything)

Every arm run since §2a-ter left the shipped configuration unchanged. The productive
output was a measurement, not an arm: with the options treated as what they
are — **verbatim corpus claims** — payload quality gets a model-free definition, and it
dominates the definition the project has used all week.

| rivals in payload | n | recorded accuracy |
|---|---|---|
| 0 | 17 | **0.706** |
| 1 | 32 | 0.250 |
| 2 | 21 | **0.143** |

**How many candidate answers we serve spans 56 points; whether the right one is present
spans 31.** AUC against the recorded answer: `1/purity` 0.741, `hit` 0.666. Purity is
computable at serve time from the `options` the platform already sends, which makes it
the first serve-time signal that survived after every score magnitude failed to predict
anything (abstention AUCs 0.40-0.56). It is not yet a lever — see below — but it is the
first quantity in this campaign that carries information about the answer rather than
about resemblance.

**The obvious rule built on it is wrong in the sign, and that is the section's other
result.** "Keep the leading seat, drop a second seat that introduces a rival claim" was
tested by recording which seat holds the answer. In the one-rival class gold owns seat 1
**6/32** times, sits later **8** times, is absent entirely **18** times, and a rival owns
seat 1 **62.5 %** of the time. The rule would discard the only present answer more often
than it keeps it. The reason is structural and already documented twice: seat 1 is chosen
by the four score terms, and those terms cannot rank gold above the winner. Any rule that
trusts the ranking inherits a judgement known to be wrong more often than right — which
is worth checking **before** building a rule of that kind in future.

**Closed today, each for the cost of one local build and zero answer calls:** label
artefact (under the undisputable claim label gold is still the pool's best chunk only
21.4 % of the time, so the finding is real and not the label talking), provenance
(a canary — the answer's own sentence injected into a distractor session — is served
88 % of the time, so the pipeline is content-seeking and location is not the obstacle),
granularity (+42 % memory count, `hit` unmoved), the conflict rule above, and the
L3 card (refuted twice, plus never emittable). **The card spend was not incurred.**

**What is left, stated precisely and not optimistically:** the 40 questions where the
answer session is absent score **0.175**, below the 0.243 no-memory floor — we are worse
than useless there, and withholding would be worth +0.039. **That +0.039 is now the
weakest number in this file** (see the collapse arm below): a payload with every
candidate-answer sentence removed still left its answer-absent half at 0.150, nine points
under the prior, so the deficit is not caused by the claim text the payload carries and
an empty payload has no measured claim to collect it. No score magnitude detects it
(abstention AUCs 0.40-0.56), and purity does not either: it counts answers served, not
whether the right one is among them. **What turned out detectable at serve time is
conflict, not presence** — `scripts/option_match_calib.py` shows an item-to-option
matcher separates perfectly, because a true match is self-similarity (all 101 positives
exactly 1.0000; worst of 2 583 negatives 0.9677), and at tau 0.97 the per-query option
count agrees with the verbatim label on 70/70, firing on exactly the 35 conflicted
queries. So **conflict-collapse** — when the payload commits to two or more options, drop
the claim-bearing items rather than trust any of them — is buildable and is the one
untried candidate with a mechanism in its favour. Its value could not be read off the
pooled comparison, and the reason is worth keeping: the no-conflict state it manufactures
**occurs naturally 5 times at the shipped budget**, and across five arms the class
*shrinks* as budget grows (17/5/4/3/3 at ms 1/2/3/4/6) while its accuracy rises
monotonically over that same axis (0.353→1.000) — a selection signature, not an effect.
Splitting the fired set instead, by whether the answer would be destroyed, does bound it:

| ms=2 partition at tau 0.97 | n | accuracy |
|---|---|---|
| not fired | 35 | 0.457 |
| fired, answer among the dropped items | 15 | 0.333 |
| fired, answer absent anyway | **20** | **0.100** |

| projection (answer-present → , answer-absent → ) | whole set |
|---|---|
| both end at the observed harm level 0.175 | −1.3 pp |
| both fall back to the no-memory floor 0.243 | **+2.1 pp** |
| answer-absent reaches the comparable natural state 0.600 | +12.3 pp |

The floor is positive mechanically, not speculatively: those 20 queries sit at **0.100**,
fourteen points below what the model does with no memory at all, so removing an item that
is actively steering the answer wrong only has to be *neutral* to pay. One question
survives all of this and cannot be answered locally: **whether the residual payload is
neutral.** That is the arm, and it was not run — the 350 calls are a deliberate
deferral, not an oversight, alongside the tau band's 3.2 % margin (valid only while
options are verbatim, so it must be re-validated against any question set that
paraphrases). The wider evaluation-design point stands underneath it: an instrument whose
distractors do not quote corpus claims would break the reach/confusion symmetry that makes
presence unknowable from resemblance — but that, and this arm, are next-cycle material.

**The arm was then run (350 calls, same day) and it loses: 0.329 → 0.300, +1/−3,
p=0.625, CI [−0.086, +0.029].** The mechanism is not what failed. The gate fired exactly
where the calibration said it would (fired payloads 9.86 → 7.83 items, the 35
unfired queries' payloads byte-identical at 9.31, no payload ever emptied), and the score
moved the wrong way:

| partition | n | ms=2 | collapse |
|---|---|---|---|
| not fired — the control, pure answer-model noise | 35 | 0.457 | 0.429 (+0/−1) |
| fired, gold claim aboard | 15 | 0.333 | 0.200 (+0/−2) |
| fired, no answer to destroy | 20 | 0.100 | **0.150** (+1/−0) |

Two things die here, and only one of them was a hypothesis. **The rescue row does not
exist**: the 20 answer-absent queries went to 0.150, a single question, less than the
control group drifted by accident, so their 0.100 was never an actively misleading item
steering the model away from a 24 % baseline — it is absence, and subtracting a rival
cannot add an answer. **The optimistic row was impossible as written**, and I wrote it
anyway: in a fired query the gold option is verbatim corpus text, so its carrying item
matches the gold option at self-similarity 1.0 and the rule drops it. Any rule that
subtracts claim-bearing items from a payload that holds the answer trades two-fifths of a
correct answer for a coin flip, and that bound was computable from the rule's definition
before a single call was spent. Incidental and not a gain: with the claims stripped the
answer model got *more* certain (unanimous 0.971 vs 0.929, per-pass spread
0.300–0.300 vs 0.314–0.343) and no more accurate.

`conflict_collapse` stays off, reproducible behind `--conflict-collapse`. This is the
end of the purity axis as a serve-time *action*: conflict is the signal inside purity, it
is detectable at 70/70, and acting on it by subtraction costs 2.9 pp. What is left of the
0.175/0.243 gap is index-side — deciding not to build a two-option payload at all, rather
than building one and removing from it — and the eval-design point underneath it (an
instrument whose distractors do not quote corpus claims) is next-cycle material with
everything else in this file.

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
| 1 | Smoke test passed | **open** — needs the deployed endpoint; every track self-hosts since 2026-09-26. **Calendar: evaluation closes 2026-11-04 23:59; one Full run takes 0.5–2 days wall clock — deploy and smoke early enough to leave room for one retry.** During the soak: the answer-call error rate under real network, and (if select-llm is enabled via env) the `session selection failed` warning rate |
| 2 | API contract correct | covered: `tests/test_contract.py` (identifier echo, durability-then-success, top_k ceiling, error envelope, auth, isolation, idempotency, concurrency) |
| 3 | Add/Search models = `gpt-4o-mini` | the shipped graded path uses no LLM (`session_select_llm` reverted to off — §2); when enabled via env it adds one selection call per Search (`CODEMEM_LLM_MODEL`, default `gpt-4o-mini`, `max_tokens=24` — choices only, never returned text); the dense/rerank encoders are non-generative, disclosed in `docs/COMPLIANCE.md` |
| 4 | ≥30 days publicly reachable | **open — the long pole**: VM direct port exposure per `deploy/README.md` (CDN/edge proxies cap timeouts below the worst-case Add and are recorded as unsafe) |
| 5 | Run instructions complete | `README.md` (quick start, API, config), `deploy/README.md` (image, ingress), `.env.example` (every flag, including `CODEMEM_SESSION_SELECT_LLM` and `CODEMEM_SESSION_SELECT_TIMEOUT_SECONDS`) |
| 6 | Originality disclosed | method changes are itemised in `eval/README.md` (each with its measurement); the paper survey supporting the design is part of the submission materials |
| 7 | Substantive | non-trivial measured method with negative results documented |
| 8 | No manipulation | Search never generates returned text (`tests/test_traceability.py`); the selection stage emits session choices only (`max_tokens=24`; guard tests `test_llm_select_reorders_sessions` and the fused-order stack guard in `tests/test_session_features.py`); `user_id` isolation on every read/write path (`tests/test_isolation.py`); the proxy benchmark is labelled not-the-scored-suite in its own metadata |

## 5. Parked (with reasons)

- **~~Admission-side recall work~~ — closed 2026-10-07 by `scripts/claim_bottleneck.py`,
  kept here so it is not re-opened.** It was queued as the last untried surface after
  abstention closed. It measured zero headroom: the decisive sentence is in the pool and
  admitted for 70/70 claim-anchor questions, so an `entity_timeline` eligibility channel
  has no failure class to attack (`entity_timeline` / `supersedes` stay inactive, which is
  now consistent with evidence rather than with neglect). What remains untried is narrower
  and honestly labelled: the 13 questions where correct content was served and the model
  still answered wrong are an answer-model problem, and the redistribution result above
  says no presentation knob on this axis turns them.
- **Item-level interleaving of the seated sessions — run 2026-10-07 and rejected on its
  own pre-registered gate.** `evidence_session_interleave` (off, `CODEMEM_SESSION_
  INTERLEAVE`) slots seated sessions round-robin so a later session's head reaches the
  full-form window. Gate: decidable up AND ambiguity not up, measured on the
  deterministic instrument before any relay spend. It failed at both budgets: at @1 000
  tokens, 0.600 → 0.567 with ambiguity 0.067 → 0.200 (two sessions) and 0.633 → 0.567
  with 0.100 → 0.267 (three). So the shipped session-blocking is load-bearing —
  one candidate answer in front at a time is what holds the dominant error mode down,
  the same reason `evidence_operative_promotion` is capped at 2. Zero answer calls were
  spent on it, which is what the pre-check is for. The shipped path was verified
  byte-identical with the flag off (153 tests; the recorded arm reproduces to the digit),
  not assumed.
- **~~Issue-language card overview~~ — closed 2026-10-08, refuted twice.** It had been
  justified by "the answer session reaches the top-8 for only 22/70" — a figure that is
  itself stale: the F1-selection re-run showed the menu is 97.1 % saturated under the
  option-aware plans the platform actually sends, so that 22/70 was a bare-plan artefact
  and the recall case for a card never held. Independently, the mechanism a card was
  supposed to supply — a compact unit carrying the claim — was built for free by splitting
  prose finer (95.6 → 135.2 memories per session, +42 %) and `hit` did not move (0.371-0.400,
  where one question is 1.4 pp), on top of the 2026-09-26 measurement that cards leave
  `answer_session_shown` unchanged at 43.8 %. A card also can never be emitted (invariant 1,
  rule §8). Do not spend the ~300 Add + ~350 answer calls.
- **~~Budget × gate interaction, `evidence_max_sessions` 3–4~~ — the budget half is done
  2026-10-08; the gate half is not.** ms 1/2/3/4/6 priced end-to-end on the claim anchor:
  0.314 / **0.329** / 0.314 / 0.243 / 0.257 against reach 22.9 → 94.3 %. The axis is a
  conflict-versus-presence dial whose product peaks at the shipped 2, and ms=4 sits exactly
  on the no-memory floor. What remains unpriced is `dense_eligible@0.45` crossed with the
  gate, which needs the frozen 58-question procedure instrument and therefore a cycle
  boundary.
- **Sealed holdout usage**: the claim instrument's 35 sealed questions are
  spent only on a final confirmation of whatever ships next cycle — one
  run, pre-registered, no tuning against them. (The instrument-growth
  item itself is no longer parked: the second anchor was built 2026-10-05
  and immediately earned its keep by reversing §2.)
