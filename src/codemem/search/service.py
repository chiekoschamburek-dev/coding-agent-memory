"""Search orchestration.

Reads only ``user_id``-scoped memory and never generates content: this pipeline
plans probes, recalls from several channels, fuses, scores, gates, and formats.
The LLM (when enabled, from P3) is permitted only to *score* existing memories;
it never writes text that is returned to the platform.
"""

from __future__ import annotations

import re
import time

from ..core.config import Settings
from ..core.logging import get_logger
from ..core.tokens import count_tokens
from ..index.store import MemoryRow, Store
from .evidence import (
    INFORMATIVE_CHANNELS,
    EvidenceItem,
    _iso_from_ms,
    _select_span,
    assemble,
    score_candidates,
    session_terms,
    fused_session_order,
)
from .query import plan_query
from .retriever import Retriever

log = get_logger("codemem.search")


def _ordered_blocks(reranked, memories, ordering, rest_after=None):
    """Sort candidates by session-block order, first appearance preserved
    within a block. Sessions absent from ``ordering`` follow, in their
    relative order. (``rest_after`` is accepted for call-site readability;
    absent sessions already sort after every listed one.)"""
    block = {sid: i for i, sid in enumerate(ordering)}
    pos = {id(c): i for i, c in enumerate(reranked)}

    def key(cand):
        memory = memories.get(cand.memory_id)
        sid = memory.session_id if memory else None
        if sid in block:
            return (block[sid], pos[id(cand)])
        return (len(block) + pos[id(cand)], pos[id(cand)])

    return sorted(reranked, key=key)


class SearchPipeline:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        embedder=None,
        reranker=None,
        listwise=None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = Retriever(settings, store, embedder=embedder)
        self.reranker = reranker
        self.listwise = listwise

    def handle(
        self, *, user_id: str, query: str, options: list[str] | None, top_k: int
    ) -> list[EvidenceItem]:
        t0 = time.monotonic()
        limit = max(1, min(top_k, self.settings.max_top_k))

        plan = plan_query(query, options)
        plan = self._hyde_probe(plan)
        candidates = self.retriever.recall(user_id, plan)
        if not candidates:
            log.info(
                "search empty",
                extra={"ctx": {"user_id": user_id, "reason": "no_candidates"}},
            )
            return []

        memories = self.retriever.load(user_id, candidates)

        # Exact-identifier evidence is computed per chunk by the store (IDF
        # weighted), then projected onto the memories that cover those chunks.
        chunk_scores = self.store.entity_match_scores(user_id, plan.entities)
        entity_match_by_memory: dict[int, float] = {}
        if chunk_scores:
            for cand in candidates:
                memory = memories.get(cand.memory_id)
                if memory is None or memory.chunk_id is None:
                    continue
                score = chunk_scores.get(memory.chunk_id, 0.0)
                if score > 0:
                    entity_match_by_memory[memory.id] = score

        scored = score_candidates(
            candidates,
            memories,
            plan,
            entity_match_by_memory,
            dense_only_min_similarity=(
                self.settings.dense_eligible_min_similarity
                if self.settings.dense_eligible
                else None
            ),
            dense_only_max=self.settings.dense_eligible_max,
            operative_weight=self.settings.operative_rank_weight,
        )

        # Rerank the head of the fused list by reading each (question, memory)
        # pair jointly. Only the top slice is reranked: a cross-encoder is
        # O(pool) forward passes and the tail could not reach the answer model
        # anyway.
        reranked = self._rerank(user_id, plan, scored, memories)

        # Final stage: an LLM judges the head of the list comparatively. A
        # cross-encoder scores each pair in isolation, so it cannot tell that
        # forty candidates all come from the same repository and only one
        # explains the failure; a listwise read can. Scores only -- no model
        # text is ever returned (see the module docstring).
        reranked = self._listwise_rerank(plan, reranked, memories)

        # Fusion feeds the shortlist: when enabled, the fused ordering is
        # applied to the reranked list FIRST, so the LLM's top-8 — and the
        # fallback order — are the fused ones (the diagnostic measured 20 of
        # 31 select-llm misses as shortlist-bottleneck). One stage owns the
        # final ordering: the LLM's picks when it succeeds, the fused order
        # otherwise.
        session_features = (
            self._session_features(user_id, plan, reranked, memories, query)
            if self.settings.session_feature_fusion else None
        )
        if session_features is not None:
            f1, rare_cov = session_features
            head_order: list[str] = []
            for cand in reranked:
                memory = memories.get(cand.memory_id)
                sid = memory.session_id if memory else None
                if sid and sid not in head_order:
                    head_order.append(sid)
            fused_order = fused_session_order(head_order, f1, rare_cov)
            reranked = _ordered_blocks(reranked, memories, fused_order)
            session_features = None  # applied; assemble must not re-fuse

        select_order = (
            self._llm_select_sessions(user_id, plan, reranked, memories, query)
            if self.settings.session_select_llm else None
        )
        if select_order:
            # The LLM picked the two sessions that record the cause or fix;
            # their blocks go first (in its preference order) and the
            # assembler runs unchanged — the noise gate still applies to every
            # session and the per-session caps are untouched.
            reranked = _ordered_blocks(reranked, memories, select_order, rest_after=len(select_order))

        items = assemble(
            self.settings,
            self.store,
            user_id,
            plan,
            reranked,
            memories,
            top_k=limit,
            session_features=session_features,
        )
        items = self._dense_fill(user_id, plan, candidates, memories, items)
        items = self._claim_fill(user_id, plan, memories, items)
        items = self._collapse_conflicts(items, options)

        log.info(
            "search done",
            extra={
                "ctx": {
                    "user_id": user_id,
                    "intent": plan.intent,
                    "top_k": limit,
                    "returned": len(items),
                    "tokens": sum(i.tokens for i in items),
                    "reranked": bool(self.reranker and reranked is not scored),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                }
            },
        )
        return items

    def _session_features(
        self,
        user_id: str,
        plan: QueryPlan,
        reranked: list,
        memories: dict,
        query: str,
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """The two session-level signals for the rank fusion, or None.

        F1: cosine between the query and each candidate session's FIRST raw
        message — the issue statement lives at the trajectory head, written in
        the vocabulary an issue-like query uses, while the chunks that carry
        the session's score are in tool vocabulary. F2: how much of the
        query's rare vocabulary (terms covered by at most two candidate
        sessions) the session's pooled chunks cover as a union — invisible to
        the per-chunk max the session score uses. Costs one embed call over
        the candidate sessions' first messages; never raises.
        """
        if not self.settings.session_feature_fusion:
            return None
        try:
            order: list[str] = []
            members: dict[str, list] = {}
            for cand in reranked:
                memory = memories.get(cand.memory_id)
                if memory is None:
                    continue
                if memory.session_id not in members:
                    order.append(memory.session_id)
                    members[memory.session_id] = []
                members[memory.session_id].append(memory)
            order = order[:40]
            if not order:
                return None

            f1: dict[str, float] = {sid: 0.0 for sid in order}
            embedder = self.retriever.embedder
            if embedder is not None and embedder.available:
                firsts = self.store.first_messages(user_id, order)
                live = [sid for sid in order if firsts.get(sid)]
                vectors = embedder.embed([query] + [firsts[sid] for sid in live]) or []
                if vectors and len(vectors) == len(live) + 1:
                    qvec = vectors[0]
                    for sid, svec in zip(live, vectors[1:]):
                        if len(qvec) == len(svec):
                            f1[sid] = sum(a * b for a, b in zip(qvec, svec))

            query_terms = session_terms(query)
            unions = {
                sid: set().union(*(session_terms(m.text) for m in members[sid]))
                if members[sid]
                else set()
                for sid in order
            }
            rare = {
                t for t in query_terms
                if sum(1 for sid in order if t in unions[sid]) <= 2
            }
            rare_cov = {
                sid: (len(unions[sid] & rare) / len(rare) if rare else 0.0)
                for sid in order
            }
            return f1, rare_cov
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "session features skipped",
                extra={"ctx": {"user_id": user_id, "error": str(exc)[:200]}},
            )
            return None

    _SELECT_SYSTEM = (
        "You select which past engineering sessions recorded the cause or the "
        "fix of a described problem. Reply with exactly two numbers."
    )

    def _hyde_probe(self, plan):
        """Query-side HyDE: one relay call rewrites the query as the note a
        past session would have recorded; that text rides along as an extra
        probe (lexical BM25s every probe; dense embeds query + first three).
        The claim-anchor funnel measured the shipped plan seating the answer
        session in the top-8 for 22/70 questions while 36 more were admissible
        at rank 9+ — a representation gap between how questions arrive (raw
        issue) and how answers are stored (recorded cause-prose). With the
        probe, the offline replay moved the menu to 29/70 (net +7, 9 wins /
        2 losses). Fail-safe by construction: no relay, timeout, or empty
        reply returns the plan unchanged. Entity channel untouched, so the
        effect attributes to lexical + dense only.
        """
        if not self.settings.hyde_probe:
            return plan
        if not (self.settings.llm_base_url and self.settings.llm_api_key):
            return plan
        try:
            issue = plan.query.split("Issue:\n", 1)[-1].split(
                "\n\nWhich statement", 1
            )[0].strip() if "Issue:\n" in plan.query else plan.query
            note = self._chat(
                "You write the note an engineering session would have recorded "
                "while diagnosing or fixing a problem. Reply with the note only: "
                "first person, 1-3 sentences, naming concrete identifiers "
                "(functions, files, flags), stating the cause or the fix.",
                issue[:2400],
            )
            if note and note.strip():
                plan.probes = plan.probes + [note.strip()]
        except Exception:  # never fail Search on the probe
            return plan
        return plan

    def _llm_select_sessions(
        self,
        user_id: str,
        plan: QueryPlan,
        reranked: list,
        memories: dict,
        query: str,
    ) -> list[str] | None:
        """One LLM call over compact session summaries; never raises.

        Returns the two chosen session ids in preference order, or None when
        the relay fails or the reply does not parse — the caller then keeps
        the shipped ordering. The judgement is the one the platform's Answer
        model makes ("can this context answer this question"), made over
        session-level summaries rather than the single chunks the failed
        listwise stage scored.
        """
        if not self.settings.llm_base_url or not self.settings.llm_api_key:
            return None
        try:
            order: list[str] = []
            members: dict[str, list] = {}
            for cand in reranked:
                memory = memories.get(cand.memory_id)
                if memory is None:
                    continue
                if memory.session_id not in members:
                    order.append(memory.session_id)
                    members[memory.session_id] = []
                members[memory.session_id].append(memory)
            order = order[:8]
            if len(order) < 2:
                return None

            firsts = self.store.first_messages(user_id, order)
            blocks = []
            for i, sid in enumerate(order, start=1):
                texts = [m.text for m in members[sid]]
                files = sorted({
                    f for t in texts
                    for f in re.findall(r"[\w/\\.-]+\.\w{1,4}\b", t)
                })[:6]
                top_chunks = sorted(texts, key=lambda t: -len(t))[:2]
                opening = (firsts.get(sid) or texts[0] if texts else "")[:280]
                blocks.append(
                    f"[{i}] files: {', '.join(files) if files else '(none)'}\n"
                    f"    opening: {opening}\n"
                    + "\n".join(f"    chunk: {t[:200]}" for t in top_chunks)
                )
            user = (
                "Problem / issue:\n" + query[:800] + "\n\n"
                "Candidate sessions:\n" + "\n".join(blocks) + "\n\n"
                "Which TWO sessions record the cause or the fix of this "
                "problem? Reply with the two numbers."
            )
            # Self-consistency: the digest replay measured 8 of the 11
            # recorded judgment misses flipping on a single re-call at
            # temperature 0 — those queries sit on the decision boundary and
            # the one-shot pick samples cross-request variance. With
            # votes > 1 the picks are tallied across calls and the mode
            # wins; with votes = 1 this is the shipped one-shot path
            # unchanged (first-seen order preserves the reply's preference).
            votes = max(1, int(self.settings.session_select_votes))
            tally: dict[str, int] = {}
            first_seen: dict[str, int] = {}
            for _ in range(votes):
                reply = self._chat(self._SELECT_SYSTEM, user)
                if not reply:
                    continue
                numbers = [int(n) for n in re.findall(r"\d+", reply)]
                one = [order[n - 1] for n in numbers if 1 <= n <= len(order)]
                one = list(dict.fromkeys(one))[:2]
                if len(one) < 2:
                    continue
                for pos, sid in enumerate(one):
                    first_seen.setdefault(sid, pos)
                    tally[sid] = tally.get(sid, 0) + 1
            picked = sorted(
                tally, key=lambda s: (-tally[s], first_seen[s])
            )[:2]
            if len(picked) < 2:
                return None
            log.info(
                "session selection",
                extra={"ctx": {
                    "user_id": user_id,
                    "picked": [p[:12] for p in picked],
                    "baseline_head": order[0][:12],
                }},
            )
            return picked
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "session selection failed",
                extra={"ctx": {"user_id": user_id, "error": str(exc)[:200]}},
            )
            return None

    def _chat(self, system: str, user: str) -> str | None:
        """One chat completion against the configured relay. Never raises.

        The selection call runs on its own hard budget
        (``session_select_timeout_seconds``): the relay answered in ~1 s when
        healthy, and the fallback to the shipped ordering only exists if it
        fires before the caller gives up on Search.
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.session_select_timeout_seconds,
            )
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=24,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "selection LLM call failed",
                extra={"ctx": {"error": str(exc)[:200]}},
            )
            return None

    _CLAIM_SELECT_SYSTEM = ""  # selection is embedding-only; no relay call

    def _claim_fill(
        self,
        user_id: str,
        plan: QueryPlan,
        memories: dict,
        items: list,
    ) -> list:
        """Claims-only dense side-channel: APPEND claim-shaped memories that
        the option probes retrieve above an absolute cosine floor.

        Non-competing by construction: the append happens after assembly and
        never revisits or displaces what the seated sessions carry — the
        anti-churn constraint from three measured non-monotonicities. The
        claim index is the user's assistant cause-prose (a shape filter over
        stored memories), with vectors reused from the dense channel; no
        relay call. Never raises.
        """
        settings = self.settings
        if not settings.claim_channel or not items:
            return items
        embedder = self.retriever.embedder
        if embedder is None or not embedder.available or not plan.probes:
            return items
        try:
            index = self._claim_index(user_id)
            if not index:
                return items

            probe_vecs = embedder.embed(plan.probes[:4]) or []
            if not probe_vecs:
                return items

            hits = []
            for pvec in probe_vecs:
                if not pvec:
                    continue
                for entry in index:
                    vec = entry["vec"]
                    if not vec or len(vec) != len(pvec):
                        continue
                    score = sum(a * b for a, b in zip(pvec, vec))
                    if score >= settings.claim_channel_floor:
                        hits.append((score, entry))
            hits.sort(key=lambda pair: (-pair[0], pair[1]["memory_id"]))

            tail = items[-1].score
            out = list(items)
            used_memory = {item.memory_id for item in items}
            used_text = {item.content for item in items}
            attached = 0
            for score, entry in hits:
                if attached >= settings.claim_channel_cap:
                    break
                if entry["memory_id"] in used_memory:
                    continue
                if entry["text"] in used_text:
                    continue
                nxt = tail - 1e-5
                if nxt < 1e-6:
                    break
                tail = nxt
                out.append(
                    EvidenceItem(
                        memory_id=entry["memory_id"],
                        content=entry["text"],
                        score=round(nxt, 6),
                        created_at=entry["created_at"],
                        tokens=count_tokens(entry["text"]),
                        truncated=False,
                        superseded=entry["superseded"],
                    )
                )
                used_memory.add(entry["memory_id"])
                used_text.add(entry["text"])
                attached += 1
            if attached:
                log.info(
                    "claim fill",
                    extra={"ctx": {"user_id": user_id, "attached": attached}},
                )
            return out
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "claim fill skipped",
                extra={"ctx": {"user_id": user_id, "error": str(exc)[:200]}},
            )
            return items

    def _claim_index(self, user_id: str) -> list[dict]:
        """Lazily built claims-only index for one user: assistant cause-prose
        memories with their stored dense vectors. Cached on the instance with
        the user's memory count as the staleness token (a later Add rebuilds)."""
        cached = getattr(self, "_claim_index_cache", None)
        total = self.store.user_memory_count(user_id)
        if cached and cached.get("user_id") == user_id and cached.get("total") == total:
            return cached["index"]
        rows = self.store.claim_rows(user_id)
        vecs = self.store.vectors_for(user_id, [r["memory_id"] for r in rows])
        index = [
            {
                "memory_id": r["memory_id"],
                "session_id": r["session_id"],
                "text": r["text"],
                "created_at": r["created_at"],
                "superseded": r["superseded"],
                "vec": vecs.get(r["memory_id"]),
            }
            for r in rows
            if r["memory_id"] in vecs and self._claim_shape(r["text"])
        ]
        self._claim_index_cache = {
            "user_id": user_id, "total": total, "index": index,
        }
        return index

    _CAUSE = re.compile(
        r"\b(because|caused by|due to|which (?:causes?|fails?|leads? to|makes? it)|"
        r"root cause|the problem is|the issue is|"
        r"(?:does not|doesn't) handle|fails? when|instead of|rather than|"
        r"must (?:also|be|set|be updated)|requires? (?:that|both|updating|the)|"
        r"in order to|otherwise|silently (?:ignored|fails?)|needed to|"
        r"so that|turns out|the fix (?:was|is)|which means|"
        r"this (?:happens|breaks|works|way)|we need to|have to|"
        r"was (?:because|caused)|results? (?:in|from)|resulting in|"
        r"the reason|fix(?:ed)? by|workaround|make sure|"
        r"without (?:this|that)|with this change|it should|the cause)\b",
        re.IGNORECASE,
    )
    _CLAIM_SYMBOL = re.compile(r"`[^`]{2,}`|[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*|\w+\.\w+")
    _CLAIM_SCAFFOLD = re.compile(
        r"swebench_|testbed/|my (?:mock|fake|test)\w*|manual\.yaml|_preds\.json|scratch",
        re.IGNORECASE,
    )

    def _claim_shape(self, text: str) -> bool:
        text = text or ""
        if not (80 <= len(text) <= 700):
            return False
        if text.lstrip().startswith(("[tool", "```", "[result]")):
            return False
        if not self._CAUSE.search(text) or not self._CLAIM_SYMBOL.search(text):
            return False
        return not self._CLAIM_SCAFFOLD.search(text)

    def _collapse_conflicts(self, items: list, options: list[str] | None) -> list:
        """Drop claim-bearing items when the payload commits to two or more options.

        Rejected by its own arm (350 calls, 2026-10-08): 0.329 -> 0.300 on the claim
        anchor, +1/-3, p=0.625. The gate did what the calibration promised - it fired on
        the 35 conflicted queries and nowhere else, dropping 2.03 items per fired payload
        and touching no unfired one - and the score still fell, because the loss was
        never a rival steering the model away from an answer that was aboard. On the 20
        fired queries with no answer in the payload, removing every claim left them at
        0.150 rather than at the 0.243 no-memory prior the projection assumed, a single
        question better than the baseline and less than the unfired control moved by
        accident. On the 15 where the answer was aboard it destroyed the answer by
        construction: the gold option is verbatim corpus text, so its item matches it at
        1.0 and cannot survive this rule. 0.333 -> 0.200.

        Kept off and kept working, because the arm that closed the purity axis is not
        reproducible without it, and because the conservative edge is the part of the
        rule that was never tested: if every item carries a claim the payload stands,
        since collapsing to nothing is abstention and abstention is unmeasured.
        """
        settings = self.settings
        if not settings.conflict_collapse or not items or not options:
            return items
        embedder = self.retriever.embedder
        if embedder is None or not embedder.available:
            return items
        try:
            clean = [o.strip() for o in options if o and o.strip()]
            if len(clean) < 2:
                return items
            vectors = embedder.embed(clean + [item.content for item in items]) or []
            if len(vectors) != len(clean) + len(items):
                return items
            ovec, ivec = vectors[: len(clean)], vectors[len(clean):]
            tau = settings.conflict_collapse_tau
            carriers: list[tuple[int, ...]] = []
            matched: set[int] = set()
            for ivector in ivec:
                hits = tuple(
                    oi
                    for oi, ovector in enumerate(ovec)
                    if sum(a * b for a, b in zip(ovector, ivector)) >= tau
                )
                carriers.append(hits)
                matched.update(hits)
            if len(matched) < 2:
                return items
            kept = [item for item, hits in zip(items, carriers) if not hits]
            if not kept:
                return items
            log.info(
                "conflict collapse",
                extra={"ctx": {"dropped": len(items) - len(kept), "options": len(matched)}},
            )
            # Kept items are a subsequence of a strictly decreasing score sequence,
            # so the contract's "higher score means more relevant" still holds.
            return kept
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "conflict collapse skipped",
                extra={"ctx": {"error": str(exc)[:200]}},
            )
            return items

    def _dense_fill(
        self,
        user_id: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        memories: dict[int, MemoryRow],
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        """Append a few memories that only the dense channel reached.

        The append-only form of dense-only admission, deliberately weaker than
        scoring them normally: nothing the lexical channels produced is revisited.
        Their order, their scores and the session slots they occupy all stay
        exactly as ``assemble`` decided.

        Three exclusions define what may be appended:

        * not already returned;
        * not from a session already represented — a second chunk of a session
          already shown costs tokens without adding a new lead;
        * not contested by ``lexical`` or ``entity``. That last one is what makes
        this a *fill* for the lexical channels' blind spot rather than a second
        chance for something they already ranked and left behind.

        It never fires on an empty result: the noise gate having abstained is a
        decision about the query, and appending to ``[]`` would undo it.

        Scores continue strictly decreasing from the tail, since emission order
        is the relevance order we assert.
        """
        settings = self.settings
        if not settings.dense_fill or not items or settings.dense_fill_max <= 0:
            return items

        floor = settings.dense_fill_min_similarity
        used_memory = {item.memory_id for item in items}
        used_session: set[str] = set()
        for item in items:
            memory = memories.get(item.memory_id)
            if memory is not None and memory.session_id:
                used_session.add(memory.session_id)

        fillable = [
            cand
            for cand in candidates
            if cand.memory_id not in used_memory
            and not any(name in cand.channels for name in INFORMATIVE_CHANNELS)
            and cand.channel_scores.get("dense", 0.0) >= floor
        ]
        fillable.sort(key=lambda c: (-c.channel_scores.get("dense", 0.0), c.memory_id))

        out = list(items)
        tail = out[-1].score if out else 0.0
        for cand in fillable:
            if len(out) - len(items) >= settings.dense_fill_max:
                break
            memory = memories.get(cand.memory_id)
            # A card never reaches the payload through any path: its text is
            # generated, and data[].content must stay a verbatim span of Add
            # input (docs/DESIGN.md, card invariant 1).
            if (
                memory is None
                or memory.kind == "card"
                or memory.session_id in used_session
            ):
                continue
            content, truncated = _select_span(
                memory.text,
                settings.dense_fill_tokens,
                plan.keywords,
                settings.evidence_operative_weight,
            )
            if not content:
                continue
            # Stay below the last emitted score by more than the 6-decimal
            # rounding can absorb, or two neighbours collapse to one value.
            # Stop rather than emit a duplicate at the floor: `assemble` clamps
            # to 1e-6, so a long tail can leave no room left to decrease into.
            nxt = tail - 1e-5
            if nxt < 1e-6:
                break
            tail = nxt
            out.append(
                EvidenceItem(
                    memory_id=memory.id,
                    content=content,
                    score=round(tail, 6),
                    created_at=_iso_from_ms(memory.ts) or memory.created_at,
                    tokens=count_tokens(content),
                    truncated=truncated,
                    superseded=memory.superseded_by is not None,
                )
            )
            used_memory.add(memory.id)
            if memory.session_id:
                used_session.add(memory.session_id)
        return out

    def _rerank_body(self, memory: MemoryRow, plan: QueryPlan) -> str:
        """The text the cross-encoder reads for one memory.

        A character prefix is the wrong window on a long trajectory: it keeps
        whichever lines happen to come first, not the ones that match, and the
        listwise experiment already showed the judge gets better when it sees
        more of the relevant excerpt. With ``rerank_span_tokens`` set, the
        window is chosen the same way the returned item's content is — by
        query-term density with operative lines weighted — so the stage scores
        the part of the memory that is actually on topic.
        """
        budget = self.settings.rerank_span_tokens
        if budget <= 0:
            return memory.text[: self.settings.rerank_max_chars]
        span, _ = _select_span(
            memory.text,
            budget,
            plan.keywords,
            self.settings.evidence_operative_weight,
        )
        return span

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Map cross-encoder output onto the 0..1 scale the blend expects.

        Most cross-encoders emit an unbounded logit, which a fixed temperature
        turns into a contribution that does not depend on the pool. Some (the
        bge-reranker family) emit a 0..1 relevance score already; putting those
        through the same sigmoid flattens every candidate towards 0.5 and
        destroys the ordering, so they are taken as they come.
        """
        if self.settings.rerank_probability_scores:
            return [min(1.0, max(0.0, s)) for s in scores]
        from ..rerank import sigmoid

        temperature = self.settings.rerank_temperature
        return [sigmoid(s / temperature) for s in scores]

    def _rerank(
        self,
        user_id: str,
        plan: QueryPlan,
        scored: list[Candidate],
        memories: dict[int, MemoryRow],
    ) -> list[Candidate]:
        """Blend cross-encoder scores into the fused ranking, in place.

        Returns ``scored`` unchanged when reranking is disabled or unavailable,
        so the pipeline degrades to the recall ordering.
        """
        if not scored:
            return scored
        if self.settings.rerank_session_level:
            return self._rerank_sessions(plan, scored, memories)
        if not self.reranker:
            return scored
        head = scored[: self.settings.rerank_top_n]
        texts: list[str] = []
        for cand in head:
            memory = memories.get(cand.memory_id)
            if memory is None:
                texts.append("")
                continue
            # Include the deterministic header fields so the cross-encoder sees
            # the identifiers too, not only prose.
            texts.append(
                f"{memory.structural_kind}: {self._rerank_body(memory, plan)}"
            )

        scores = self.reranker.score(plan.query, texts)
        if not scores or len(scores) != len(head):
            return scored

        # Temperature-scaled sigmoid, NOT max-normalised.
        #
        # Normalising by the maximum made every score depend on which items
        # happened to be in the reranked head, so changing rerank_top_n changed
        # the ranking non-monotonically (measured: top_n=30 scored MRR 0.789,
        # top_n=60 scored 0.772, top_n=120 scored 0.818 — an incoherent
        # sequence). A fixed temperature keeps the mapping absolute: the same
        # logit always yields the same contribution, whatever the pool size.
        normalized = self._normalize_scores(scores)

        weight = self.settings.rerank_weight
        for cand, rerank_score in zip(head, normalized):
            cand.final = (1.0 - weight) * cand.final + weight * rerank_score

        ranked = sorted(scored, key=lambda c: (-c.final, c.memory_id))
        top_final = ranked[0].final if ranked else 0.0
        if top_final > 0:
            for cand in ranked:
                cand.final = cand.final / top_final
        return ranked

    def _rerank_sessions(
        self,
        plan: QueryPlan,
        scored: list[Candidate],
        memories: dict[int, MemoryRow],
    ) -> list[Candidate]:
        """Score one representative document per session, not per entry.

        The entry-level stage spends its whole budget inside whichever sessions
        happen to fill the pool — 1.3 of them on the proxy corpus — so it can
        only re-order chunks within a session. Judging one document per session
        spends the same number of forward passes on the question the payload
        actually asks: which session comes first.

        The score is applied to every candidate of that session, so the
        within-session order that ``assemble`` preserves is left intact and only
        the session order moves.
        """
        if not self.reranker:
            return scored

        head = scored[: self.settings.rerank_top_n]
        groups: dict[str, list[Candidate]] = {}
        order: list[str] = []
        loose: list[Candidate] = []
        for cand in head:
            memory = memories.get(cand.memory_id)
            session = memory.session_id if memory is not None else None
            if session is None:
                loose.append(cand)
                continue
            if session not in groups:
                groups[session] = []
                order.append(session)
            groups[session].append(cand)

        def document(cand: Candidate) -> str:
            memory = memories.get(cand.memory_id)
            if memory is None:
                return ""
            return f"{memory.structural_kind}: {self._rerank_body(memory, plan)}"

        # ``scored`` is sorted, so a session's first entry is its strongest.
        # That is the representative the cross-encoder reads.
        targets: list[list[Candidate]] = [[cand] for cand in loose]
        texts = [document(cand) for cand in loose]
        for session in order:
            group = groups[session]
            texts.append(document(group[0]))
            targets.append(group)

        scores = self.reranker.score(plan.query, texts)
        if not scores or len(scores) != len(targets):
            return scored

        normalized = self._normalize_scores(scores)

        weight = self.settings.rerank_weight
        for group, rerank_score in zip(targets, normalized):
            for cand in group:
                cand.final = (1.0 - weight) * cand.final + weight * rerank_score

        ranked = sorted(scored, key=lambda c: (-c.final, c.memory_id))
        top_final = ranked[0].final if ranked else 0.0
        if top_final > 0:
            for cand in ranked:
                cand.final = cand.final / top_final
        return ranked

    def _listwise_rerank(
        self,
        plan: QueryPlan,
        scored: list[Candidate],
        memories: dict[int, MemoryRow],
    ) -> list[Candidate]:
        """Blend LLM listwise relevance into the ranking, in place.

        Returns ``scored`` unchanged when listwise reranking is disabled or
        unavailable, so the pipeline falls back to the fused/cross-encoder order.

        The question text sent to the judge is the platform's query, and the
        documents are stored memory text. Nothing the model produces is returned:
        only the parsed scores are used, which keeps Search on the right side of
        the "return memory evidence, do not generate" rule.
        """
        if not self.listwise or not scored:
            return scored

        head = scored[: self.settings.listwise_max_candidates]
        documents: list[str] = []
        for cand in head:
            memory = memories.get(cand.memory_id)
            documents.append(memory.text if memory is not None else "")
        if not any(documents):
            return scored

        raw_scores = self.listwise.score(plan.query, documents)
        if not raw_scores:
            return scored

        from ..listwise import normalise

        judged = normalise(raw_scores)
        if len(judged) != len(head):
            return scored

        weight = self.settings.listwise_weight
        for cand, score in zip(head, judged):
            if score is None:
                continue  # unjudged: leave the existing score alone
            cand.final = (1.0 - weight) * cand.final + weight * score

        ranked = sorted(scored, key=lambda c: (-c.final, c.memory_id))
        top = ranked[0].final if ranked else 0.0
        if top > 0:
            for cand in ranked:
                cand.final = cand.final / top
        return ranked
