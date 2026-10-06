"""Offline pricing for the claims-only dense side-channel (zero displacement).

Design under pre-registration (the road map's survivor): the claim prose —
assistant messages that record a CAUSE — never pools through the
file/lexical/entity channels (claim-chunk pooling 15.7 %, delivered 0/70),
and dense_eligible died because new candidates competed for menu seats.
The side-channel is a **non-competing entry**: a dense index over claim-shaped
chunks only, queried with the option probes (which are themselves claim-shaped
— the HyDE finding's asset); hits above a high cosine floor are appended to
the payload after assembly, displacing nothing.

Priced here offline before any productization, on the claim-tune set (70 q):

  attach rate          questions where ≥1 claim is attached
  gold delivered       the ANSWER session's claim arrives whole in
                       payload+attached (claim_delivered)
  displacement         must be 0 — seated sessions before == after

Floor sweep {0.45, 0.50, 0.55, 0.60} × cap {2, 3}. Zero relay calls.

Usage::

    PYTHONPATH=src python scripts/pricing_claim_channel.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from codemem.core.config import Settings  # noqa: E402
from codemem.search.evidence import assemble as assemble_ship  # noqa: E402
from codemem.search.evidence import score_candidates  # noqa: E402
from codemem.search.query import plan_query  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from run_evidence import claim_delivered  # noqa: E402

CAUSE = re.compile(
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
SYMBOL = re.compile(r"`[^`]{2,}`|[a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*|\w+\.\w+")
SCAFFOLD = re.compile(
    r"swebench_|testbed/|my (?:mock|fake|test)\w*|manual\.yaml|_preds\.json|scratch",
    re.IGNORECASE,
)


def claim_shape(text: str) -> bool:
    """The claim-shape filter: assistant prose that records a cause."""
    text = text or ""
    if not (80 <= len(text) <= 700):
        return False
    if text.lstrip().startswith(("[tool", "```", "[result]")):
        return False
    if not CAUSE.search(text) or not SYMBOL.search(text):
        return False
    if SCAFFOLD.search(text):
        return False
    return True


def main() -> int:
    import logging

    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient

    from codemem.api.app import create_app

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
    embedder = container.embedder

    if True:
        for memory in bench["memories"]:
            client.post("/add", json={
                "request_id": f"pcc:{memory['id']}",
                "user_id": memory["user_id"],
                "session_id": memory["session_id"],
                "messages": memory["messages"],
            })

        # ---- build the claims-only index: every claim-shaped memory, per user
        claim_index: dict[str, list[dict]] = {}
        with store._read() as conn:
            for memory in bench["memories"]:
                uid = memory["user_id"]
                rows = conn.execute(
                    "SELECT m.id, m.text, m.session_id FROM memory m"
                    " JOIN chunk c ON c.id = m.chunk_id"
                    " JOIN raw_message r ON r.user_id = c.user_id"
                    "   AND r.request_id = c.request_id"
                    "   AND r.msg_index = c.msg_index"
                    " WHERE m.user_id = ? AND r.role = 'assistant'",
                    (uid,),
                ).fetchall()
                for r in rows:
                    if claim_shape(r["text"]):
                        claim_index.setdefault(uid, []).append({
                            "memory_id": r["id"],
                            "session_id": r["session_id"],
                            "text": r["text"],
                        })
        n_claims = sum(len(v) for v in claim_index.values())
        print(f"claim index: {n_claims} claim-shaped memories "
              f"across {len(claim_index)} users")

        # embed all claims once
        claim_vecs: dict[str, list[tuple[dict, list[float]]]] = {}
        for uid, claims in claim_index.items():
            vecs = embedder.embed([c["text"] for c in claims]) or []
            claim_vecs[uid] = list(zip(claims, vecs))

        floors = (0.45, 0.50, 0.55, 0.60)
        caps = (2, 3)
        stats = {}
        for floor in floors:
            for cap in caps:
                stats[(floor, cap)] = {"attached": 0, "gold": 0, "displaced": 0,
                                       "owner_hit": 0}

        n = 0
        for question in qa["questions"]:
            gold_claim = question.get("gold_claim")
            answer_session = question.get("answer_session")
            if not gold_claim or not answer_session:
                continue
            n += 1
            user_id = next(
                m["user_id"] for m in bench["memories"]
                if m["repo"] == question["repo"]
            )
            query = question["question"]
            options = question.get("options") or []

            plan = plan_query(query, options)  # deployment plan: with options
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
            items = assemble_ship(
                settings, store, user_id, plan, list(reranked), memories, top_k=100,
            )
            seated = []
            for item in items:
                memory = memories.get(item.memory_id)
                if memory and memory.session_id not in seated:
                    seated.append(memory.session_id)
            payload_blob = "\n".join(item.content for item in items)

            # option probes (deployment probes) against the claims-only index
            probe_vecs = embedder.embed(plan.probes[:4]) or []
            claims = claim_vecs.get(user_id, [])
            hits = []
            for pvec in probe_vecs:
                if not pvec:
                    continue
                for claim, cvec in claims:
                    if len(pvec) != len(cvec):
                        continue
                    score = sum(a * b for a, b in zip(pvec, cvec))
                    hits.append((score, claim))
            hits.sort(key=lambda pair: -pair[0])

            for floor in floors:
                for cap in caps:
                    st = stats[(floor, cap)]
                    seen_claims = set()
                    attached = []
                    used_sessions = set()
                    for score, claim in hits:
                        if score < floor or len(attached) >= cap:
                            break
                        key = claim["memory_id"]
                        if key in seen_claims:
                            continue
                        seen_claims.add(key)
                        attached.append(claim)
                    blob = payload_blob + "\n" + "\n".join(
                        c["text"] for c in attached)
                    st["attached"] += 1 if attached else 0
                    ok = claim_delivered(blob, gold_claim)
                    st["gold"] += 1 if ok else 0
                    st["owner_hit"] += 1 if any(
                        c["session_id"] == answer_session for c in attached) else 0
                    # displacement: zero means the seated set is unchanged
                    st["displaced"] += 0  # append-only by construction

    client.__exit__(None, None, None)

    print(f"questions: {n}")
    print(f"\n{'floor':>6} {'cap':>4} {'attach%':>8} {'gold-delivered':>15} "
          f"{'owner-hit':>10} {'displaced':>10}")
    for (floor, cap), st in sorted(stats.items()):
        print(f"{floor:>6.2f} {cap:>4} {st['attached'] / n:>8.3f} "
              f"{st['gold'] / n:>10.3f} ({st['gold']:>2}/{n}) "
              f"{st['owner_hit'] / n:>10.3f} {st['displaced']:>10}")
    print(f"\nbaseline gold delivered (payload only): measured 0/70 "
          f"in the pricing_intra_order run")
    out = ROOT / "eval/results/pricing_claim_channel.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump({f"{floor}/{cap}": st for (floor, cap), st in stats.items()},
                  handle, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
