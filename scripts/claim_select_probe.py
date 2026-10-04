"""Why does select-llm lose on the claim-topical anchor? A one-call probe.

Phase D measured the ordering flip: max 0.300 > floor 0.243 > select 0.229
on the rebuilt claim instrument, with select reaching FEWER answer sessions
(26/70) than the deterministic top-2 (30/70). The hypothesis this probe
tests: the instrument's adversarial construction — the designated
distractor claim is MORE similar to the issue than the gold — extends to
the session level, and the selection LLM follows that semantic gradient
straight into the distractor sessions. "The selection stage inherits the
answer model's prior" cuts both ways: where the prior points at
distractors, aligning with it amplifies the error.

Per claim question: replay the pipeline (one shared Add pass), take the
top-8 menu, build the shipped digest, one relay call, then classify the
two picks against {answer session, distractor sessions, other}.

Zero answer-model calls; one relay call per question.

Usage::

    PYTHONPATH=src python scripts/claim_select_probe.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MENU = 8


def load_env() -> None:
    import os

    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    import logging
    import os

    logging.disable(logging.WARNING)
    load_env()
    base_url = os.environ.get("CODEMEM_LLM_BASE_URL")
    key = os.environ.get("CODEMEM_LLM_API_KEY")
    model = os.environ.get("CODEMEM_LLM_MODEL", "gpt-4o-mini")
    if not (base_url and key):
        print("error: relay credentials missing (.env)", file=sys.stderr)
        return 2

    from fastapi.testclient import TestClient
    from openai import OpenAI

    from codemem.api.app import create_app
    from codemem.core.config import Settings
    from codemem.search.evidence import score_candidates
    from codemem.search.query import plan_query

    bench = json.loads((ROOT / "eval/data/benchmark.json").read_text(encoding="utf-8"))
    qa = json.loads(
        (ROOT / "eval/data/qa_claim_tune.json").read_text(encoding="utf-8")
    )

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    container = app.state.container
    store = container.store
    pipeline = container.search

    for memory in bench["memories"]:
        client.post("/add", json={
            "request_id": f"cp:{memory['id']}",
            "user_id": memory["user_id"],
            "session_id": memory["session_id"],
            "messages": memory["messages"],
        })

    relay = OpenAI(base_url=base_url, api_key=key, timeout=30.0)
    system = (
        "You select which past engineering sessions recorded the cause or "
        "the fix of a described problem. Reply with exactly two numbers."
    )

    rows = []
    for question in qa["questions"]:
        user_id = next(
            m["user_id"] for m in bench["memories"] if m["repo"] == question["repo"]
        )
        query = question["question"]
        plan = plan_query(query, None)
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
        members: dict[str, list] = {}
        for cand in reranked:
            memory = memories.get(cand.memory_id)
            if memory is None:
                continue
            if memory.session_id not in members:
                order.append(memory.session_id)
                members[memory.session_id] = []
            members[memory.session_id].append(memory)
        menu = order[:MENU]
        if len(menu) < 2:
            rows.append({"query_id": question["query_id"], "menu_too_small": True})
            continue

        firsts = store.first_messages(user_id, menu)
        blocks = []
        for i, sid in enumerate(menu, start=1):
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
        try:
            response = relay.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=24,
            )
            reply = response.choices[0].message.content or ""
        except Exception as exc:  # relay failure recorded, not fatal
            rows.append({
                "query_id": question["query_id"], "relay_error": str(exc)[:120],
            })
            continue
        numbers = [int(n) for n in re.findall(r"\d+", reply)]
        picked = [menu[n - 1] for n in numbers if 1 <= n <= len(menu)]
        picked = list(dict.fromkeys(picked))[:2]

        distractors = set(question.get("distractor_sessions") or [])
        rows.append({
            "query_id": question["query_id"],
            "answer_session": question["answer_session"],
            "answer_in_menu": question["answer_session"] in menu,
            "answer_picked": question["answer_session"] in picked,
            "picked_distractors": [p for p in picked if p in distractors],
            "picked_other": [
                p for p in picked
                if p != question["answer_session"] and p not in distractors
            ],
        })

    client.__exit__(None, None, None)

    ok = [r for r in rows if "answer_session" in r]
    print(f"probed {len(ok)} questions "
          f"({sum(1 for r in rows if 'relay_error' in r)} relay errors)")
    print(f"answer session in menu:    "
          f"{sum(r['answer_in_menu'] for r in ok)}/{len(ok)}")
    print(f"answer session picked:     "
          f"{sum(r['answer_picked'] for r in ok)}/{len(ok)}")
    in_menu = [r for r in ok if r["answer_in_menu"]]
    if in_menu:
        print(f"  of those in menu, picked: "
              f"{sum(r['answer_picked'] for r in in_menu)}/{len(in_menu)}")
    picked_any_d = sum(1 for r in ok if r["picked_distractors"])
    print(f"picks that landed on a question's distractor sessions: "
          f"{picked_any_d}/{len(ok)}")
    total_d = sum(len(r["picked_distractors"]) for r in ok)
    total_picks = sum(
        len(r["picked_distractors"]) + len(r["picked_other"]) + r["answer_picked"]
        for r in ok
    )
    print(f"total picks: {total_picks}; on distractors: {total_d}; "
          f"on answer: {sum(r['answer_picked'] for r in ok)}")

    out = ROOT / "eval/results/claim_select_probe.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
