"""Query-side HyDE: generate the hypothetical claim, replay the menu.

Pre-registered in eval/README.md ("Query-side HyDE"). Two phases, both
cached where they cost money:

  1. generation — one gpt-4o-mini call per tune question rewrites the
     issue as the statement a past session would have recorded; cached in
     eval/results/hyde_claims.json (reruns never re-spend relay calls);
  2. replay — the shipped plan vs the plan with the HyDE text appended as
     an extra probe (lexical: every probe is BM25'd; dense: query + first
     three probes are embedded), staged exactly like claim_funnel.py:
     menu / rank9+ / gated / not_pooled, with per-query menu win/loss.

Kill line 1: the HyDE plan must gain >= 3 NET menu seats (wins minus
losses) for the e2e arm to be built.

Usage::

    PYTHONPATH=src python scripts/hyde_probe.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import (  # noqa: E402
    INFORMATIVE_CHANNELS,
    score_candidates,
)
from codemem.search.query import plan_query  # noqa: E402

MENU = 8
CACHE = ROOT / "eval/results/hyde_claims.json"

SYSTEM = (
    "You write the note an engineering session would have recorded while "
    "diagnosing or fixing a problem. Reply with the note only: first "
    "person, 1-3 sentences, naming concrete identifiers (functions, "
    "files, flags), stating the cause or the fix."
)


def load_env() -> None:
    import os

    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def issue_of(question: str) -> str:
    body = question.split("Issue:\n", 1)[-1]
    return body.split("\n\nWhich statement", 1)[0].strip()


def generate(question: str, model: str, base_url: str, key: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=key, timeout=30.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question[:2400]},
        ],
        temperature=0,
        max_tokens=160,
    )
    return (response.choices[0].message.content or "").strip()


def main() -> int:
    import logging
    import os

    logging.disable(logging.WARNING)
    load_env()
    base_url = os.environ.get("CODEMEM_LLM_BASE_URL")
    key = os.environ.get("CODEMEM_LLM_API_KEY")
    model = os.environ.get("CODEMEM_LLM_MODEL", "gpt-4o-mini")

    bench = json.loads((ROOT / "eval/data/benchmark.json").read_text(encoding="utf-8"))
    claim = json.loads(
        (ROOT / "eval/data/qa_claim_tune.json").read_text(encoding="utf-8")
    )

    # ---- phase 1: generation (cached) -------------------------------------
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    spent = 0
    for question in claim["questions"]:
        qid = question["query_id"]
        if qid in cache:
            continue
        try:
            cache[qid] = generate(issue_of(question["question"]), model,
                                  base_url, key)
        except Exception as exc:
            print(f"  generation failed {qid}: {str(exc)[:100]}")
            cache[qid] = ""
        spent += 1
    if spent:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    ok = sum(1 for v in cache.values() if v)
    print(f"hyde claims: {ok}/{len(claim['questions'])} "
          f"({spent} new relay calls)")

    # ---- phase 2: replay ---------------------------------------------------
    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    container = app.state.container
    store = container.store
    pipeline = container.search

    for memory in bench["memories"]:
        client.post("/add", json={
            "request_id": f"hp:{memory['id']}",
            "user_id": memory["user_id"],
            "session_id": memory["session_id"],
            "messages": memory["messages"],
        })

    def stage(user_id: str, plan, answer: str) -> tuple[str, int | None]:
        candidates = pipeline.retriever.recall(user_id, plan)
        memories = pipeline.retriever.load(user_id, candidates)
        chunk_scores = store.entity_match_scores(user_id, plan.entities)
        entity_match = {}
        for cand in candidates:
            memory = memories.get(cand.memory_id)
            if memory is None or memory.chunk_id is None:
                continue
            score = chunk_scores.get(memory.chunk_id, 0.0)
            if score > 0:
                entity_match[memory.id] = score
        scored = score_candidates(
            candidates, memories, plan, entity_match,
            operative_weight=settings.operative_rank_weight,
        )
        reranked = pipeline._rerank(user_id, plan, scored, memories)  # noqa: SLF001
        order: list[str] = []
        for cand in reranked:
            m = memories.get(cand.memory_id)
            if m is None:
                continue
            if m.session_id not in order:
                order.append(m.session_id)
        pos = order.index(answer) if answer in order else None
        if pos is None:
            pooled = {
                m.session_id for c in candidates
                if (m := memories.get(c.memory_id)) is not None
                and any(n in c.channels for n in INFORMATIVE_CHANNELS)
            }
            all_pooled = {
                m.session_id for c in candidates
                if (m := memories.get(c.memory_id)) is not None
            }
            if answer in pooled:
                return "gated", None
            if answer in all_pooled:
                return "gated", None  # pooled but nothing gate-passing
            return "not_pooled", None
        return ("menu" if pos < MENU else "rank9+"), pos

    rows = []
    for question in claim["questions"]:
        qid = question["query_id"]
        user_id = next(
            m["user_id"] for m in bench["memories"] if m["repo"] == question["repo"]
        )
        answer = question["answer_session"]
        base_plan = plan_query(question["question"], None)
        hyde_plan = plan_query(question["question"], None)
        if cache.get(qid):
            hyde_plan.probes = hyde_plan.probes + [cache[qid]]
        base_stage, base_pos = stage(user_id, base_plan, answer)
        hyde_stage, hyde_pos = stage(user_id, hyde_plan, answer)
        rows.append({
            "query_id": qid,
            "base": base_stage, "base_pos": base_pos,
            "hyde": hyde_stage, "hyde_pos": hyde_pos,
        })

    client.__exit__(None, None, None)

    for name in ("base", "hyde"):
        counts = Counter(r[name] for r in rows)
        print(f"\n{name}: {dict(counts)}")

    menu_win = [r for r in rows if r["hyde"] == "menu" and r["base"] != "menu"]
    menu_loss = [r for r in rows if r["base"] == "menu" and r["hyde"] != "menu"]
    net = len(menu_win) - len(menu_loss)
    print(f"\nmenu: base {sum(1 for r in rows if r['base'] == 'menu')}/70, "
          f"hyde {sum(1 for r in rows if r['hyde'] == 'menu')}/70")
    print(f"menu wins {len(menu_win)}, losses {len(menu_loss)}, net {net:+d}")
    for r in menu_win:
        print(f"  + {r['query_id']:48s} {r['base']:11s} -> menu (pos {r['hyde_pos']})")
    for r in menu_loss:
        print(f"  - {r['query_id']:48s} menu -> {r['hyde']:11s}")
    gate_fixed = [r for r in rows if r["base"] == "gated" and r["hyde"] != "gated"]
    print(f"gated bucket resolved: {len(gate_fixed)}/11 "
          f"({Counter(r['hyde'] for r in gate_fixed)})")
    print(f"\nKILL LINE 1 (net >= +3 menu seats): "
          f"{'PASS - e2e arm justified' if net >= 3 else 'FAIL - dead, no e2e'}")

    out = ROOT / "eval/results/hyde_probe.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "net_menu": net}, handle, ensure_ascii=False,
                  indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
