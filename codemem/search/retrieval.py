"""Search pipeline: cards only.

Order of operations, and why:

1. Scope first -- ``user_id`` is resolved before any query is issued, and every
   store call is scoped, so "search globally then filter" cannot happen.
2. Card-only multi-channel recall, each with its own quota, then *rank-based*
   fusion. Fusion is RRF over ranks rather than a weighted sum of raw scores,
   because BM25, cosine and entity-hit counts are not calibrated against each
   other. The channels are: card field BM25, card entities, case anchors and
   the card's own embedding (``card_vector``); options run as a de-weighted
   probe. Chunks are never recalled as items -- they live inside cards as
   evidence.
3. Dedup and complementary-value selection against the graded token budget.
4. Applicability (soft, listwise) and external reranker, then the final cut.
5. Deterministic template rendering -- no free-form generation here.

Cards below the top item must clear a deterministic floor relative to the
response's best score.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from .applicability import (
    ApplicabilityJudge,
    NullApplicabilityJudge,
    fuse_with_applicability,
)
from ..common.casebook import CasePacket, build_packet
from ..common.config import Settings, get_settings
from ..add.experience import card_from_payload, render_experience
from .query import QueryFeatures, build_query_features
from .rerank import NullReranker, Reranker
from ..common.store import Store
from ..common.tokens import TokenCounter, get_counter
from .transferability import TransferCard, transferability_hint

# --------------------------------------------------------------------------
# result model
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    item_id: str
    kind: str  # "experience"
    text: str
    channels: dict[str, int] = field(default_factory=dict)  # channel -> 1-based rank
    fused_score: float = 0.0
    payload: dict | None = None

    def channel_rank(self, channel: str) -> int:
        return self.channels.get(channel, 0)


class ExperienceVectorIndex(Protocol):
    """Semantic recall over cards, returning experience ids."""

    @property
    def available(self) -> bool: ...

    def search(self, user_id: str, query: str, limit: int) -> list[tuple[str, float]]: ...


class NullExperienceVectorIndex:
    available = False

    def search(self, user_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        return []


class StoreExperienceVectorIndex:
    """Card vector channel backed by the store's ``experience_vectors`` table.

    Every failure returns ``[]``, because the vector channel may only add
    candidates -- it can never break lexical recall or turn a successful Add
    into an error.  This is the only embedding store: chunks are never
    embedded.
    """

    def __init__(self, store: Store, embedder, model: str, timeout_s: float | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.model = model
        self.timeout_s = timeout_s
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        try:
            return bool(self.embedder is not None and self.embedder.available)
        except Exception:
            return False

    def search(self, user_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        if not self.available or not query.strip():
            return []
        try:
            result = self.embedder.embed([query], timeout_s=self.timeout_s)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        if result is None or not result.vectors:
            self.last_error = "embedding_failed"
            return []
        try:
            return self.store.experience_vector_search(
                user_id, result.vectors[0], self.model, limit
            )
        except Exception as exc:
            self.last_error = f"experience_vector_search:{type(exc).__name__}"
            return []


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    channels: dict[str, list[tuple[str, float]]],
    weights: dict[str, float],
    k: int,
) -> dict[str, float]:
    """RRF over per-channel ranks.

    Using ranks sidesteps the fact that BM25 magnitudes, cosine similarities
    and entity hit counts have no common scale. Weights are channel-level
    priority multipliers; they are *not* a re-introduction of raw score mixing.
    """
    fused: dict[str, float] = {}
    for channel, ranked in channels.items():
        weight = weights.get(channel, 1.0)
        if weight <= 0:
            continue
        for rank, (item_id, _score) in enumerate(ranked, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + rank)
    return fused


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


@dataclass
class SearchDiagnostics:
    channels: dict[str, int] = field(default_factory=dict)
    pool_size: int = 0
    after_dedup: int = 0
    selected: int = 0
    dropped_duplicate: int = 0
    dropped_budget: int = 0
    dropped_subzero_budget: int = 0
    tokens_used: int = 0
    token_mode: str = "estimate"
    degraded: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    applicability: dict = field(default_factory=dict)
    rerank: dict = field(default_factory=dict)
    dropped_final: int = 0
    patch_suppressed: int = 0
    dropped_low_score: int = 0
    # Optional request deadline; when present the engine skips the vector
    # channel rather than let an embedding call overrun the search budget.
    deadline: object | None = None


class SearchEngine:
    def __init__(
        self,
        store: Store,
        settings: Settings | None = None,
        token_counter: TokenCounter | None = None,
        judge: ApplicabilityJudge | None = None,
        reranker: Reranker | None = None,
        card_vector_index: ExperienceVectorIndex | None = None,
        transferability_judge=None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.card_vector_index = card_vector_index or NullExperienceVectorIndex()
        self.counter = token_counter or get_counter(self.settings)
        self.judge = judge or NullApplicabilityJudge()
        self.reranker = reranker or NullReranker()
        self.transferability_judge = transferability_judge

    # -- public ----------------------------------------------------------

    def search(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        options: list[str] | None = None,
        deadline=None,
    ) -> tuple[list[dict], SearchDiagnostics]:
        diagnostics = SearchDiagnostics(token_mode=self.counter.mode)
        diagnostics.deadline = deadline
        features = build_query_features(
            query, options, denoise=self.settings.query_denoise
        )
        final_limit = max(
            0, min(top_k, self.settings.top_k_max, self.settings.search_max_items)
        )
        if final_limit == 0:
            return [], diagnostics
        response_limit = final_limit

        pool = self._recall_cards(user_id, features, diagnostics)
        diagnostics.pool_size = len(pool)

        ordered = self._rank(pool, features)
        diagnostics.after_dedup = len(ordered)

        selected = self._select(ordered, diagnostics, response_limit)
        if self.settings.transferability_enabled:
            selected = self._apply_transferability(
                selected, features, diagnostics, deadline, user_id=user_id,
            )
            diagnostics.rerank = {"applied": False, "reason": "replaced_by_transferability"}
        else:
            selected = self._apply_applicability(
                selected, features, diagnostics, deadline
            )
            selected = self._apply_rerank(selected, features, diagnostics, deadline)
        selected = self._finalize(selected, diagnostics, response_limit)

        # Final output stage. Two distinct budgets apply, and conflating them is
        # a bug:
        #
        #   token_budget_total  -- the HARD cap. The response must never exceed
        #                          it. This is what the output stage enforces.
        #   token_budget_target -- the SOFT target. It drives marginal-value
        #                          selection in ``_select`` (stop padding once
        #                          past it), but it must never truncate output:
        #                          a high-value item may legitimately push past
        #                          the soft target.
        #
        # A candidate that does not fit the remaining hard budget is compressed
        # to whatever room is left, and if even that is not enough it is skipped
        # so a later, shorter candidate can still be included -- never a hard
        # stop on the whole output.
        hard_cap = self.settings.token_budget_total
        min_body_tokens = self.settings.min_output_body_tokens
        data: list[dict] = []
        accumulated_tokens = 0

        # Admission, not padding. RRF scores are not comparable across queries
        # (they depend on how many channels fired and at which ranks), so the
        # floor is relative to this response's best item. A query that matches
        # nothing has top_score == 0 and admits only the ``min_output_items``
        # guarantee -- returning five "least bad" cards is what put 3.75/5
        # same-repo noise into the response.
        top_score = max(
            (candidate.fused_score for candidate in selected), default=0.0
        )
        if top_score > 0.0:
            floor = max(
                self.settings.min_score_abs,
                top_score * max(0.0, self.settings.min_score_ratio),
            )
        else:
            floor = math.inf
        min_items = max(0, self.settings.min_output_items)
        emitted = 0

        for rank, candidate in enumerate(selected, start=1):
            card = (candidate.payload or {}).get("card")
            is_minimal = bool(card and card.is_minimal)
            # An idea-level card is not a repair packet padded into the
            # response: it must earn its slot over the same floor.  The
            # min-output guarantee stays for legacy packet cards.
            if (is_minimal or emitted >= min_items) and candidate.fused_score < floor:
                diagnostics.dropped_low_score += 1
                continue
            remaining = hard_cap - accumulated_tokens
            if remaining <= 0:
                diagnostics.dropped_budget += 1
                continue

            # The applicability warning is part of the returned content, so its
            # tokens must be reserved *before* rendering.  Appending it after a
            # cap-sized render could push the item over `remaining` and make the
            # output stage drop it -- i.e. the warning would delete the very
            # evidence it is warning about.
            hint = _applicability_hint(candidate)
            hint_tokens = self.counter.count(hint) if hint else 0

            # Experience items carry a self-contained three-section render and
            # get their own, wider per-item cap: the embedded evidence quotes
            # count toward this item's tokens.
            tier_cap = self.settings.experience_per_item_token_cap(rank)
            cap = max(0, min(tier_cap, remaining) - hint_tokens)
            allow_patch = _patch_allowed(candidate, features)
            if not allow_patch:
                diagnostics.patch_suppressed += 1
            evidence = () if is_minimal else self._card_evidence(
                candidate.payload["card"], features,
                k=self.settings.card_evidence_items,
            )
            content = render_experience(
                candidate.payload["card"],
                candidate.payload.get("chunks_by_id", {}),
                self.counter,
                cap,
                self.settings.experience_quote_max_chars,
                allow_patch=allow_patch,
                evidence=evidence,
                patch_render=self.settings.experience_patch_render,
                include_chunk_provenance=False,
                persisted=True,
            )
            score = round(candidate.fused_score, 6)

            if not content.strip():
                diagnostics.dropped_subzero_budget += 1
                continue

            # Check the evidence body *before* adding the warning, so a warning
            # can never be mistaken for evidence.
            if _body_tokens(content, self.counter) < min_body_tokens:
                diagnostics.dropped_budget += 1
                continue

            if hint:
                content = content + hint
            item_tokens = self.counter.count(content)
            # Skip (do not stop) an item whose *rendered* content would not fit:
            # returning a header with no evidence is worse than returning the
            # next, shorter candidate.
            if item_tokens > remaining:
                diagnostics.dropped_budget += 1
                continue

            accumulated_tokens += item_tokens
            emitted += 1
            entry: dict = {
                "id": candidate.item_id,
                "content": content,
                "score": score,
            }
            # Grouped order and RRF scores are different signals. The optional
            # score must not contradict the judge's final response order.
            if diagnostics.applicability.get("mode") == "transferability" and diagnostics.applicability.get("applied"):
                entry.pop("score")
            created_at = candidate.payload.get("created_at")
            if created_at:
                entry["created_at"] = created_at
            data.append(entry)

        diagnostics.selected = len(data)
        diagnostics.tokens_used = accumulated_tokens
        return data, diagnostics

    # -- recall ----------------------------------------------------------

    def _recall_cards(
        self,
        user_id: str,
        features: QueryFeatures,
        diagnostics: SearchDiagnostics,
    ) -> dict[str, Candidate]:
        """Card-only recall: every current card is a candidate, chunks never are.

        Channels order cards; they cannot introduce a chunk candidate.  The
        card's own embedding (``card_vector``, over its index face) is the
        channel that lets a card be recalled by what it *means* rather than by
        which terms it shares.
        """
        pool: dict[str, Candidate] = {}
        quota = self.settings.pool_per_channel
        query_terms = features.content_tokens or features.tokens

        channels: dict[str, list[tuple[str, float]]] = {}
        channels["experience_field"] = self.store.experience_field_search(
            user_id, query_terms, quota, features.anchors
        )
        channels["experience_entity"] = self.store.experience_entity_search(
            user_id, features.entity_values(), quota
        )
        card_vector_hits = self._card_vector_hits(
            user_id, features, diagnostics, quota
        )
        if card_vector_hits:
            channels["card_vector"] = card_vector_hits
        anchor_map = _anchor_kinds(features)
        if anchor_map:
            channels["case_anchor"] = self.store.case_anchor_search(
                user_id, anchor_map, quota
            )
        # Options are a de-weighted probe of what was *asked*.
        if features.options_text:
            option_terms = features.option_tokens or None
            channels["options"] = self.store.experience_field_search(
                user_id,
                option_terms if option_terms else features.options_text,
                quota,
                None,
            )
            option_entities = features.option_entity_values()
            if option_entities:
                channels["options_entity"] = self.store.experience_entity_search(
                    user_id, option_entities, quota
                )

        for name, ranked in channels.items():
            diagnostics.channels[name] = len(ranked)
        diagnostics.channels["card_vector"] = len(card_vector_hits)

        weights = {
            "experience_field": self.settings.fusion_weight_experience_field,
            "experience_entity": self.settings.fusion_weight_experience_entity,
            "card_vector": self.settings.fusion_weight_card_vector,
            "case_anchor": self.settings.fusion_weight_case_anchor,
            "options": self.settings.fusion_weight_options,
            "options_entity": self.settings.fusion_weight_options,
        }
        fused = reciprocal_rank_fusion(channels, weights, self.settings.rrf_k)
        for eid in self.store.list_current_experience_ids(
            user_id, self.settings.card_recall_limit
        ):
            fused.setdefault(eid, 0.0)
        if not fused:
            return pool

        ranks: dict[str, dict[str, int]] = {}
        for name, ranked in channels.items():
            for rank, (item_id, _score) in enumerate(ranked, start=1):
                ranks.setdefault(item_id, {}).setdefault(name, rank)

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        ordered = ordered[: self.settings.card_recall_limit]
        self._add_experience_candidates(
            user_id,
            [eid for eid, _ in ordered],
            ranks,
            dict(ordered),
            pool,
        )
        return pool

    def _card_vector_hits(
        self,
        user_id: str,
        features: QueryFeatures,
        diagnostics: SearchDiagnostics,
        quota: int,
    ) -> list[tuple[str, float]]:
        """Semantic recall over the card layer.

        Empty when embeddings are not configured -- the card pool then relies on
        its lexical channels, which is the documented lex-only baseline.
        """
        deadline = getattr(diagnostics, "deadline", None)
        if deadline is not None and getattr(deadline, "expired", False):
            diagnostics.degraded.append("card_vector_skipped_deadline")
            return []
        if not self.card_vector_index.available:
            return []
        try:
            return self.card_vector_index.search(
                user_id, features.query_text, quota
            )
        except Exception:
            diagnostics.degraded.append("card_vector_unavailable")
            return []

    def _add_experience_candidates(
        self,
        user_id: str,
        exp_ids: list[str],
        ranks: dict[str, dict[str, int]],
        score_by_id: dict[str, float],
        pool: dict[str, Candidate],
    ) -> None:
        """Materialise experience candidates and their embedded evidence.

        Each card carries the chunks it cites, so the output stage can render a
        self-contained three-section item without a second round trip. All
        referenced chunks are fetched once for the whole pool.
        """
        rows = self.store.fetch_current_experiences(user_id, exp_ids)
        if not rows:
            return
        cards: dict[str, tuple[object, dict]] = {}
        ref_ids: set[str] = set()
        for eid, row in rows.items():
            try:
                card = card_from_payload(json.loads(row["payload"]))
            except Exception:
                continue
            cards[eid] = (card, row)
            ref_ids.update(card.source_chunk_ids)
        ref_chunks = self.store.fetch_chunks(user_id, list(ref_ids)) if ref_ids else {}
        for eid, (card, row) in cards.items():
            stable_id = card.stable_id
            payload = {
                "card": card,
                "chunks_by_id": ref_chunks,
                "created_at": card.created_at or row.get("created_at"),
            }
            # Files are cheap to expose for MMR; full packets are built lazily
            # for the selected items only (Search is O(candidates) otherwise).
            if getattr(card, "has_packet_fields", False):
                payload["files"] = list(card.files)
            pool[stable_id] = Candidate(
                item_id=stable_id,
                kind="experience",
                # The stored default render is the cost/fingerprint estimate for
                # selection; the output stage re-renders to the actual per-item
                # cap so deeper ranks compress without losing 历史经验.
                text=row.get("full_content", ""),
                channels=ranks.get(eid, {}),
                fused_score=score_by_id.get(eid, 0.0),
                payload=payload,
            )

    def _card_evidence(self, card: object, features: QueryFeatures, k: int = 3) -> tuple[str, ...]:
        """Second-stage chunk rerank inside one recalled card.

        The card is the primary unit (it decides *which* historical case); its
        chunks are only evidence.  We score the card's own source chunks and,
        when those are sparse, the session's chunks, then return the top-k
        snippets without exposing chunks as independent top-level items.
        """
        if k <= 0:
            return ()
        ids = list(getattr(card, "source_chunk_ids", ()) or ())
        session_id = getattr(card, "session_id", "") or ""
        if len(ids) < k and session_id:
            for chunk in self.store.fetch_session_chunks(
                getattr(card, "user_id", "") or "", session_id, limit=400
            ):
                ids.append(chunk.chunk_id)
        ids = list(dict.fromkeys(ids))[:400]
        if not ids:
            return ()
        chunks = self.store.fetch_chunks(getattr(card, "user_id", "") or "", ids)
        query_tokens = {token.lower() for token in getattr(features, "tokens", ()) if token}
        anchors = [
            str(value).lower()
            for value in getattr(features, "anchor_values", list)()
            if str(value).strip()
        ]
        # Evidence must be executable/test evidence, not the issue prose.  The
        # issue statement is already the card's problem; repeating it as
        # "evidence" only adds noise and tokens.
        allowed_kinds = {"code", "command", "test_result", "stack", "log", "diff"}
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for chunk in chunks.values():
            if chunk.kind not in allowed_kinds:
                continue
            text = chunk.text.strip()
            if len(text) < 40:
                continue
            low = text.lower()
            lexical = sum(1 for token in query_tokens if token in low)
            exact = sum(1 for anchor in anchors if anchor and anchor in low)
            strong = bool(re.search(r"(?m)^\s*(FAILED|E\s+|\d+\s+(passed|failed))", text))
            if not exact and not strong and lexical < 2:
                continue
            score = lexical + 4.0 * exact + (2.0 if strong else 0.0)
            score /= 1.0 + len(text) / 2000.0
            fingerprint = re.sub(r"\s+", " ", low)[:80]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            scored.append((score, text))
        scored.sort(key=lambda item: -item[0])
        return tuple(text[:200] for _score, text in scored[:k])

    # -- rank ------------------------------------------------------------

    def _rank(
        self,
        pool: dict[str, Candidate],
        features: QueryFeatures,
    ) -> list[Candidate]:
        """Deterministic adjustments over fused rank, calibrated to RRF.

        The additive terms below are normalised factors multiplied by
        ``rank_boost_scale`` (default 0.004).  RRF itself spans roughly
        0.002–0.016 per channel, so these adjustments can reorder neighbours
        without becoming the ranking function.

        Outcome / verification status is deliberately NOT a ranking input. A
        "测试通过" card gets no global boost over a failed attempt -- failed
        experience is often more valuable for avoiding a dead end, and
        verification only matters as an applicability feature once the
        conditions already match.
        """
        scale = self.settings.rank_boost_scale
        for candidate in pool.values():
            factor = 0.0
            if candidate.channel_rank("experience_entity") > 0:
                factor += 1.0 / (1.0 + math.log1p(candidate.channel_rank("experience_entity")))
            if candidate.channel_rank("case_anchor") > 0:
                factor += 1.2 / (1.0 + math.log1p(candidate.channel_rank("case_anchor")))
            if candidate.channel_rank("experience_field") > 0:
                factor += 0.7 / (1.0 + math.log1p(candidate.channel_rank("experience_field")))
            # An item reached only through the options channel is weaker
            # evidence about what was *asked*; keep it but below query hits.
            if (
                candidate.channel_rank("experience_field") == 0
                and candidate.channel_rank("experience_entity") == 0
                and candidate.channel_rank("case_anchor") == 0
                and candidate.channel_rank("card_vector") == 0
            ):
                factor -= 1.5
            # A self-contained case packet is the preferred output unit when we
            # have one; a packet carrying a change/verification is worth more
            # than a problem-only card.
            if candidate.kind == "experience":
                factor += 0.4
                card = (candidate.payload or {}).get("card")
                if card is not None and getattr(card, "patch", None):
                    factor += 0.8
                elif card is not None and getattr(card, "files", ()):
                    factor += 0.3
            candidate.fused_score += scale * factor

        return sorted(pool.values(), key=lambda c: (-c.fused_score, c.item_id))

    # -- selection -------------------------------------------------------

    def _select(
        self,
        ordered: list[Candidate],
        diagnostics: SearchDiagnostics,
        final_limit: int,
    ) -> list[Candidate]:
        """Dedup + complementary-value selection for a wider L2 candidate set.

        Near-duplicates must not stack, and selection stops when marginal value
        runs out.  The result is intentionally wider than the response when a
        judge or reranker needs a real choice; otherwise it is the response set.
        """
        if final_limit <= 0:
            return []
        limit = final_limit
        if self.settings.transferability_enabled:
            limit = max(limit, self.settings.transferability_max_candidates)
        if getattr(self.judge, "available", False):
            limit = max(limit, self.settings.applicability_max_candidates)
        if getattr(self.reranker, "available", False):
            limit = max(limit, self.settings.rerank_candidates)
        limit = min(self.settings.pool_total, limit)

        seen_fingerprints: set[int] = set()
        seen_files: set[str] = set()
        selected: list[Candidate] = []
        deferred: list[Candidate] = []
        total_tokens = 0

        def try_add(candidate: Candidate) -> bool:
            nonlocal total_tokens
            rank = len(selected) + 1
            cap = self.settings.experience_per_item_token_cap(rank)
            body = candidate.text

            fingerprint = _fingerprint(body)
            if fingerprint in seen_fingerprints:
                diagnostics.dropped_duplicate += 1
                return False

            # Soft budget: once past the target, require the candidate to carry
            # enough content to be worth its cost. This is the "marginal
            # information increment" rule, expressed measurably.
            #
            # Note this is the ONLY budget rule here. Precise hard-cap fitting
            # happens in the output stage, which knows each item's *rendered*
            # size and can truncate or skip.
            body_tokens = self.counter.count(body)
            if total_tokens >= self.settings.token_budget_target and body_tokens < 40:
                diagnostics.dropped_budget += 1
                return False

            seen_fingerprints.add(fingerprint)
            total_tokens += min(body_tokens, cap)
            selected.append(candidate)
            return True

        for candidate in ordered:
            if len(selected) >= limit:
                break
            # File-coverage MMR: with ~5 reads available to the agent, three
            # packets pointing at the same file are worth less than three
            # distinct hypotheses.  Duplicated-file packets are deferred, not
            # dropped.
            files = _candidate_files(candidate)
            if files and seen_files and len(selected) >= 2 and (set(files) & seen_files):
                deferred.append(candidate)
                continue
            if try_add(candidate):
                seen_files.update(files)

        for candidate in deferred:
            if len(selected) >= limit:
                break
            try_add(candidate)

        return selected

    def _finalize(
        self,
        selected: list[Candidate],
        diagnostics: SearchDiagnostics,
        final_limit: int,
    ) -> list[Candidate]:
        """L4: truncate to the response budget."""
        if final_limit <= 0:
            return []
        if len(selected) > final_limit:
            diagnostics.dropped_final += len(selected) - final_limit
            selected = selected[:final_limit]
        return selected

    # -- rerank -----------------------------------------------------------

    def _apply_rerank(
        self,
        selected: list[Candidate],
        features: QueryFeatures,
        diagnostics: SearchDiagnostics,
        deadline,
    ) -> list[Candidate]:
        """Rerank a wider candidate set before the final response cut."""
        try:
            available = bool(self.reranker.available)
        except Exception:
            available = False
        if not available:
            diagnostics.rerank = {"applied": False, "reason": "unavailable"}
            return selected
        if len(selected) < 2:
            diagnostics.rerank = {"applied": False, "reason": "too_few"}
            return selected
        candidates = selected[: self.settings.rerank_candidates]
        documents = [candidate.text for candidate in candidates]
        original_scores = {
            candidate.item_id: candidate.fused_score for candidate in selected
        }
        try:
            scores = self.reranker.rerank(
                features.denoised_text or features.raw_query,
                documents,
                deadline=deadline,
            )
            if scores is None or len(scores) != len(candidates):
                raise ValueError("reranker returned an invalid score list")
            n = len(candidates)
            alpha = self.settings.rerank_fusion_alpha
            ranked: list[tuple[float, Candidate]] = []
            for position, (candidate, raw_score) in enumerate(
                zip(candidates, scores), start=1
            ):
                prior = (n - position + 1) / n
                fused = alpha * max(0.0, min(1.0, float(raw_score))) + (
                    1.0 - alpha
                ) * prior
                ranked.append((fused, candidate))
            ranked.sort(key=lambda item: (-item[0], item[1].item_id))
        except Exception:
            # A reranker is an enhancement; malformed output or a provider
            # exception must never fail Search.
            for candidate in selected:
                candidate.fused_score = original_scores[candidate.item_id]
            diagnostics.rerank = {"applied": False, "reason": "error"}
            diagnostics.degraded.append("rerank_failed")
            return selected
        for fused, candidate in ranked:
            candidate.fused_score = float(fused)
        rest = selected[n:]
        floor = min((score for score, _candidate in ranked), default=0.0) - 0.001
        for candidate in rest:
            candidate.fused_score = floor
        diagnostics.rerank = {"applied": True, "candidates": n, "alpha": alpha}
        return [candidate for _score, candidate in ranked] + rest

    # -- applicability ----------------------------------------------------

    def _apply_transferability(self, selected, features, diagnostics, deadline, *, user_id):
        judge = self.transferability_judge
        info = {"mode": "transferability", "applied": False}
        diagnostics.applicability = info
        if self.settings.transferability_max_candidates <= 0:
            info["reason"] = "no_candidates"
            return selected
        if judge is None or not judge.available:
            info["reason"] = "unavailable"
            diagnostics.degraded.append("transferability_unavailable")
            return selected
        cards = []
        for candidate in selected:
            card = (candidate.payload or {}).get("card")
            if card is not None and card.is_minimal:
                cards.append(TransferCard(candidate.item_id, card.problem, card.mechanism))
            if len(cards) >= max(0, self.settings.transferability_max_candidates):
                break
        try:
            result = judge.judge(user_id=user_id, query=features.raw_query,
                                 cards=cards, deadline=deadline)
        except Exception:
            info["reason"] = "error"
            diagnostics.degraded.append("transferability_failed")
            return selected
        info.update({"applied": bool(result.verdicts), "reason": result.reason,
                     "calls": result.calls, "cached": result.cached,
                     "hypotheses": len(result.hypotheses), "candidates": len(cards)})
        if not result.verdicts:
            if cards:
                diagnostics.degraded.append("transferability_failed")
            return selected
        hypotheses = {h["id"]: h for h in result.hypotheses}
        counts = {}
        for candidate in selected:
            verdict = result.verdicts.get(candidate.item_id)
            if verdict is not None:
                value = dict(vars(verdict))
                value["hypothesis"] = hypotheses.get(verdict.matched_hypothesis)
                candidate.payload["transferability"] = value
                counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
        info["verdicts"] = counts
        # Stable grouping, not filtering. Unjudged/unknown/topic-related cards
        # keep baseline order between transferable and irrelevant cards.
        group = {"transferable": 0, "irrelevant": 2}
        return sorted(selected, key=lambda c: group.get(
            (c.payload or {}).get("transferability", {}).get("verdict"), 1))

    def _apply_applicability(
        self,
        selected: list[Candidate],
        features: QueryFeatures,
        diagnostics: SearchDiagnostics,
        deadline,
    ) -> list[Candidate]:
        """Listwise applicability pass over the selected case packets.

        The judge sees all candidate packets together (that is what makes it
        listwise), returns tri-state judgments, and their scores are added in
        RRF-sized units.  It is a soft signal: no candidate is dropped, and a
        missing deadline/verdict leaves the retrieval order untouched.
        """
        if not getattr(self.judge, "available", False):
            diagnostics.applicability = {"applied": False, "reason": "unavailable"}
            return selected

        packets: list[CasePacket] = []
        for candidate in selected:
            if candidate.kind != "experience":
                continue
            payload = candidate.payload or {}
            card = payload.get("card")
            if card is not None and card.is_minimal:
                # An idea-level card has no repair packet for the legacy judge.
                continue
            packet = payload.get("packet")
            if packet is None:
                card = payload.get("card")
                if card is not None:
                    packet = build_packet(
                        card,
                        list((payload.get("chunks_by_id") or {}).values()),
                        persisted=True,
                    )
            if packet is not None and packet.has_actionable_content():
                packets.append(packet)
        if len(packets) < 2:
            diagnostics.applicability = {"applied": False, "reason": "too_few_packets"}
            return selected

        verdicts = self.judge.rerank(
            query=features.denoised_text or features.raw_query,
            packets=packets,
            deadline=deadline,
        )
        if not verdicts:
            diagnostics.applicability = {"applied": False, "reason": "no_verdict"}
            return selected

        counts = {"yes": 0, "unknown": 0, "no": 0}
        for verdict in verdicts.values():
            counts[verdict.app] = counts.get(verdict.app, 0) + 1
        diagnostics.applicability = {"applied": True, **counts}
        return fuse_with_applicability(
            selected, verdicts, self.settings.applicability_weight
        )


_ANCHOR_KIND_MAP = {
    "path": "file",
    "symbol": "symbol",
    "error": "error",
    "test_name": "test_name",
    "issue": "issue",
    "config_key": "config",
}


def _anchor_kinds(features: QueryFeatures) -> dict[str, list[str]]:
    """Map query anchors onto the case-anchor index vocabulary."""
    out: dict[str, list[str]] = {}
    for kind, values in features.anchors.items():
        target = _ANCHOR_KIND_MAP.get(kind, kind)
        bucket = out.setdefault(target, [])
        for value in values:
            if value not in bucket:
                bucket.append(value)
        if kind == "path":
            # A file path also implies its two-component module prefix, which is
            # how case conditions record the affected area.
            modules = out.setdefault("module", [])
            for value in values:
                parts = [p for p in str(value).replace("\\", "/").split("/") if p]
                if len(parts) >= 2:
                    prefix = "/".join(parts[:2])
                    if prefix not in modules:
                        modules.append(prefix)
    return out


def _patch_allowed(candidate: Candidate, features: QueryFeatures) -> bool:
    """Render the historical change only when its anchors touch the query.

    A linked experience can be merely adjacent (same module, different file).
    Showing its diff then causes negative transfer: the model copies a change
    that belongs to another task.  Class names are not enough -- ``AxesGrid``
    appears in both the wrong and the right matplotlib files.  We therefore
    require a path basename or a function-like (snake_case) symbol from the
    query to occur in the card's own text/change.
    """
    card = (candidate.payload or {}).get("card")
    if card is None or not getattr(card, "patch", None):
        return True
    query_paths = {
        str(value).replace("\\", "/").rsplit("/", 1)[-1].lower()
        for value in features.anchors.get("path", ())
        if str(value).strip()
    }
    query_functions = {
        str(value).lower()
        for value in features.anchors.get("symbol", ())
        if "_" in str(value)
    }
    if not query_paths and not query_functions:
        # Only class/issue anchors: no reliable function-level signal.  Keep the
        # change only if the query is completely anchor-free (nothing to reject).
        has_any_anchor = any(features.anchors.values())
        return not has_any_anchor
    card_paths = {
        str(value).replace("\\", "/").rsplit("/", 1)[-1].lower()
        for value in getattr(card, "files", ()) or ()
        if str(value).strip()
    }
    if query_paths & card_paths:
        return True
    if query_functions:
        haystack = " ".join(
            [
                str(getattr(card, "problem", "") or ""),
                str(getattr(card, "implement", "") or ""),
                str(getattr(card, "patch", "") or ""),
                " ".join(getattr(card, "symbols", ()) or ()),
            ]
        ).lower()
        if any(name in haystack for name in query_functions):
            return True
    return False


def _candidate_files(candidate: Candidate) -> set[str]:
    payload = candidate.payload or {}
    files = payload.get("files")
    if isinstance(files, (list, tuple, set)):
        return {str(value) for value in files if value}
    packet = payload.get("packet")
    if packet is not None:
        return set(getattr(packet, "files", ()) or ())
    return set()


def _applicability_hint(candidate: Candidate) -> str:
    """Rendered warning for a negative/uncertain judgment (empty otherwise).

    The hint is content, so its token cost is reserved before rendering; it is
    never appended after the per-item cap has already been filled.
    """
    transfer_hint = transferability_hint(candidate.payload or {})
    if transfer_hint:
        return transfer_hint
    verdict = (candidate.payload or {}).get("applicability")
    if (
        candidate.kind == "experience"
        and isinstance(verdict, dict)
        and verdict.get("app") in ("no", "unknown")
        and verdict.get("reason")
    ):
        return (
            "\n适用性提示（模型判断，非事实）："
            f"{verdict['app']}：{verdict['reason']}"
        )
    return ""


def _body_tokens(content: str, counter: TokenCounter) -> int:
    """Tokens in the evidence body, excluding the provenance header.

    A rendered packet emits its header lines before the body. An item whose
    budget was consumed by the header carries no evidence, so it should be
    skipped rather than returned.
    """
    marker = "\n\n"
    idx = content.find(marker)
    body = content[idx + len(marker):] if idx != -1 else content
    return counter.count(body)


def _fingerprint(text: str) -> int:
    """Whitespace-collapsed hash, so trivially reformatted copies dedup."""
    collapsed = " ".join(text.split())
    return hash(collapsed)
