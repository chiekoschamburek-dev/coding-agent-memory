"""Output admission: a response slot is earned, never padded.

The service used to fill ``search_max_items`` slots with whatever ranked next,
so a query that matched nothing still returned five "least bad" cards -- 3.75 of
5 items were same-repository noise in the 28-target probe. Admission is now a
threshold on the candidate's own score, relative to the best score in the same
response (RRF magnitudes are not comparable across queries), with a small
minimum so a weak query is not answered with an empty list.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from codemem.common.config import Settings  # noqa: E402
from codemem.add.experience import ExperienceCard
from codemem.search.retrieval import Candidate, SearchEngine  # noqa: E402
from codemem.common.store import Store  # noqa: E402


def _engine(tmp_path: Path, **overrides) -> SearchEngine:
    base = dict(
        db_path=tmp_path / "admission.db",
        api_key=None,
        llm_base_url=None,
        llm_api_key=None,
        embed_base_url=None,
        embed_api_key=None,
        embed_model="",
        search_max_items=5,
        experience_min_items=0,
        applicability_enabled=False,
    )
    base.update(overrides)
    settings = Settings(**base)
    return SearchEngine(Store(settings), settings)


def _stub(engine: SearchEngine, scores: list[float]) -> None:
    candidates = []
    for index, score in enumerate(scores):
        text = f"card body number {index} carrying enough words to render normally"
        candidates.append(
            Candidate(
                item_id=f"experience:e:{index}",
                kind="experience",
                text=text,
                fused_score=score,
                payload={
                    "card": ExperienceCard(
                        experience_id=f"e:{index}", user_id="u1",
                        session_id="s1", problem=text, implement="a repair",
                    ),
                    "chunks_by_id": {},
                    "created_at": None,
                },
            )
        )
    engine._recall_cards = lambda *a, **k: {c.item_id: c for c in candidates}  # type: ignore[assignment]
    engine._rank = lambda pool, features: sorted(  # type: ignore[assignment]
        pool.values(), key=lambda c: -c.fused_score
    )


def test_low_score_tail_is_not_returned(tmp_path):
    engine = _engine(tmp_path)
    _stub(engine, [1.0, 0.9, 0.05, 0.04, 0.03])
    data, diag = engine.search(user_id="u1", query="anything", top_k=5)
    # floor = 0.15 * 1.0 -> only the two real hits clear it
    assert [item["id"] for item in data] == ["experience:e:0", "experience:e:1"]
    assert diag.dropped_low_score == 3


def test_ratio_zero_restores_padding(tmp_path):
    engine = _engine(tmp_path, min_score_ratio=0.0)
    _stub(engine, [1.0, 0.9, 0.05, 0.04, 0.03])
    data, diag = engine.search(user_id="u1", query="anything", top_k=5)
    assert len(data) == 5
    assert diag.dropped_low_score == 0


def test_all_zero_pool_returns_only_the_minimum(tmp_path):
    engine = _engine(tmp_path)
    _stub(engine, [0.0, 0.0, 0.0, 0.0, 0.0])
    data, diag = engine.search(user_id="u1", query="nothing matches", top_k=5)
    # Nothing clears a floor of zero, so only the min_items guarantee applies.
    assert len(data) == 2
    assert diag.dropped_low_score == 3


def test_minimum_is_configurable_and_never_exceeds_the_pool(tmp_path):
    engine = _engine(tmp_path, min_output_items=1)
    _stub(engine, [0.0, 0.0])
    data, _diag = engine.search(user_id="u1", query="x", top_k=5)
    assert len(data) == 1

    engine = _engine(tmp_path, min_output_items=9)
    _stub(engine, [0.0, 0.0])
    data, _diag = engine.search(user_id="u1", query="x", top_k=5)
    assert len(data) == 2


def test_absolute_floor_can_admit_nothing_but_the_minimum(tmp_path):
    engine = _engine(tmp_path, min_score_abs=0.5)
    _stub(engine, [1.0, 0.9, 0.4, 0.2])
    data, diag = engine.search(user_id="u1", query="x", top_k=5)
    assert [item["id"] for item in data] == ["experience:e:0", "experience:e:1"]
    assert diag.dropped_low_score == 2
