"""Conflict-collapse tests: fires only on a real conflict, never abstains, and
leaves the shipped path byte-identical when off.

The rule is a post-assembly filter: when the returned items collectively carry the
text of two or more of the sent options, drop the carrying items and keep the rest.
Two properties are worth pinning because the measurement that motivated the rule is
about a *sub*-population, and a rule that quietly turns into abstention would show up
as an accuracy change for the wrong reason:

  * it must not fire when at most one option is carried - one committed answer is the
    state that scores well (0.706 vs 0.250 with a rival present), so removing it would
    be damage, not de-conflicting;
  * it must not empty the payload - if every item carries a claim, the collapse is
    total, and that is the unmeasured abstention policy, not this one.

The encoder is stubbed with a deterministic bag-of-words cosine, so an identical string
scores 1.0 and a disjoint one 0.0 - the same exactness the shipped matcher relies on,
without loading a model.
"""

from __future__ import annotations

import math

from codemem.core.config import Settings

CLAIM_A = "The retry handler swallows the timeout and returns stale quota rows."
CLAIM_B = "The cache key omits the tenant id so two tenants share one entry."
CONTEXT = "The quota service reads its limits from the settings file at startup."
VOCAB: dict[str, int] = {}


class BagOfWordsEncoder:
    """Deterministic cosine stand-in: identical text scores 1.0, disjoint 0.0."""

    available = True

    def embed(self, texts):
        docs = [set(t.lower().split()) for t in texts]
        for d in docs:
            for w in d:
                VOCAB.setdefault(w, len(VOCAB))
        out = []
        for d in docs:
            vec = [0.0] * len(VOCAB)
            for w in d:
                vec[VOCAB[w]] = 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def make_app(tmp_path, *, collapse: bool):
    from codemem.api.app import Container

    settings = Settings(
        data_dir=tmp_path / "data",
        dense_enabled=False,
        rerank_enabled=False,
        conflict_collapse=collapse,
        conflict_collapse_tau=0.97,
        min_evidence_score=0.01,
    )
    container = Container(settings)
    container.search.retriever.embedder = BagOfWordsEncoder()
    return container


def seed(container, turns):
    from codemem.core.schemas import Message

    for i, text in enumerate(turns):
        container.add.handle(
            request_id=f"req-{i}",
            user_id="u1",
            session_id="s1",
            messages=[Message(role="assistant", timestamp=1000 + i, content=text)],
        )


def search(container, options):
    return container.search.handle(
        user_id="u1", query="quota tenant timeout handler", options=options, top_k=10
    )


def test_off_by_default_leaves_the_committed_payload_in_place(tmp_path):
    container = make_app(tmp_path, collapse=False)
    seed(container, [CONTEXT, CLAIM_A, CLAIM_B])
    items = search(container, [CLAIM_A, CLAIM_B, "something else entirely"])
    texts = " ".join(i.content for i in items)
    assert CLAIM_A in texts and CLAIM_B in texts, "the flag is off; nothing should be dropped"


def test_fires_only_when_two_options_are_carried(tmp_path):
    container = make_app(tmp_path, collapse=True)
    seed(container, [CONTEXT, CLAIM_A, CLAIM_B])
    items = search(container, [CLAIM_A, CLAIM_B, "something else entirely"])
    texts = " ".join(i.content for i in items)
    assert CLAIM_A not in texts and CLAIM_B not in texts, "both claims should be dropped"
    assert CONTEXT in texts, "non-claim context should survive"


def test_one_committed_answer_is_left_alone(tmp_path):
    container = make_app(tmp_path, collapse=True)
    seed(container, [CONTEXT, CLAIM_A])
    items = search(container, [CLAIM_A, "an option nobody recorded", "nor this one"])
    texts = " ".join(i.content for i in items)
    assert CLAIM_A in texts, "a single committed answer is the good state, not a conflict"


def test_never_abstains_when_every_item_carries_a_claim(tmp_path):
    container = make_app(tmp_path, collapse=True)
    seed(container, [CLAIM_A, CLAIM_B])
    items = search(container, [CLAIM_A, CLAIM_B])
    assert items, "collapsing to nothing is the unmeasured abstention policy, not this rule"


def test_scores_stay_strictly_decreasing_after_a_collapse(tmp_path):
    container = make_app(tmp_path, collapse=True)
    seed(container, [CONTEXT, CLAIM_A, CLAIM_B, "Quota limits are reloaded per request in the handler."])
    items = search(container, [CLAIM_A, CLAIM_B, "unrelated option"])
    assert len(items) >= 2, "expected a partially filtered payload"
    scores = [i.score for i in items]
    assert all(a > b for a, b in zip(scores, scores[1:])), scores
