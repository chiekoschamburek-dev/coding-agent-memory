"""Entry-level attribution: is the *right memory* recalled, and what decides rank?

`scripts/attribute_recall_loss.py` answers the question per session - "did any
entry of the relevant session reach stage X". That is a weak test: a session has
~96 entries and only a handful of them mention a file the task's patch touched,
so "1 in 96 reached the pool" and "the decisive one reached the pool" look the
same from the session view. This script measures the stronger question:

  A. take the entries that actually name a gold-patch file ("gold entries");
  B. follow *those* entries through pool -> admitted -> payload -> window;
  C. for the relevant sessions that lose, dump the deterministic score
     decomposition of the candidate that beat them, so the loss can be pinned to
     a term (fused rank RRF, channel strength, identifier evidence, kind bonus,
     cross-encoder) rather than guessed at.

Run:
    PYTHONPATH=src python scripts/diagnose_entry_level.py -k 10 --out eval/results/entry_level.json

To separate "the decisive chunk is not retrievable" from "the pool truncated it",
re-read the same A/B/C with a deeper pull:

    PYTHONPATH=src python scripts/diagnose_entry_level.py --depth 600 --pool 2400 \
        --out eval/results/entry_level_deep.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

import logging

logging.disable(logging.WARNING)

from codemem.add.entities import _norm_path  # noqa: E402
from codemem.api.app import Container  # noqa: E402
from codemem.core.config import Settings  # noqa: E402
from codemem.core.schemas import Message  # noqa: E402
from codemem.search.evidence import (  # noqa: E402
    INFORMATIVE_CHANNELS,
    _operative_term,
    assemble,
    score_candidates,
)
from codemem.search.query import plan_query  # noqa: E402


def build(container: Container, data: dict) -> None:
    for memory in data["memories"]:
        container.add.handle(
            request_id=f"bench:{memory['id']}",
            user_id=memory["user_id"],
            session_id=memory["session_id"],
            messages=[Message(**m) for m in memory["messages"]],
        )


def gold_memories(store, user_id: str, files: list[str]) -> tuple[set[int], dict[str, set[int]], str]:
    """(gold memory ids, by session, which entity level matched).

    A memory is "gold" for a task when it names, as an extracted identifier, a
    file that the task's patch touched. That is the entry-level version of "the
    right memory" - the session view cannot tell it from any other chunk of the
    same session.
    """
    norms = [_norm_path(f) for f in files]
    names = [f.rsplit("/", 1)[-1] for f in norms]
    with store._read() as conn:  # noqa: SLF001 - measurement only
        rows = conn.execute(
            "SELECT DISTINCT m.id, m.session_id FROM memory m"
            " JOIN chunk_entity ce ON ce.chunk_id = m.chunk_id AND ce.user_id = m.user_id"
            " WHERE m.user_id = ? AND ce.etype = 'file_path' AND ce.value_norm IN ({})".format(
                ",".join("?" for _ in norms)
            ),
            (user_id, *norms),
        ).fetchall()
        scope = "file_path"
        if not rows:
            scope = "file_name"
            rows = conn.execute(
                "SELECT DISTINCT m.id, m.session_id FROM memory m"
                " JOIN chunk_entity ce ON ce.chunk_id = m.chunk_id AND ce.user_id = m.user_id"
                " WHERE m.user_id = ? AND ce.etype = 'file_name' AND ce.value_norm IN "
                "({})".format(",".join("?" for _ in names)),
                (user_id, *names),
            ).fetchall()
    by_session: dict[str, set[int]] = {}
    for row in rows:
        by_session.setdefault(row["session_id"], set()).add(int(row["id"]))
    return {int(r["id"]) for r in rows}, by_session, scope


def analyse(container: Container, data: dict, k: int) -> list[dict]:
    settings = container.settings
    pipeline = container.search
    store = container.store

    entries_per_session: dict[str, dict[str, int]] = {}
    for memory in data["memories"]:
        with store._read() as conn:  # noqa: SLF001
            row = conn.execute(
                "SELECT count(*) AS n FROM memory WHERE user_id=? AND session_id=?",
                (memory["user_id"], memory["session_id"]),
            ).fetchone()
        entries_per_session.setdefault(memory["user_id"], {})[
            memory["session_id"]
        ] = int(row["n"]) if row else 0

    out: list[dict] = []
    for query in data["queries"]:
        if not query["relevant"]:
            continue
        user_id = next(
            m["user_id"] for m in data["memories"] if m["repo"] == query["repo"]
        )
        gold, gold_by_session, scope = gold_memories(
            store, user_id, list(query.get("files") or ())
        )

        plan = plan_query(query["query"], None)
        candidates = pipeline.retriever.recall(user_id, plan)
        memories = pipeline.retriever.load(user_id, candidates)
        chunk_scores = store.entity_match_scores(user_id, plan.entities)
        entity_match: dict[int, float] = {}
        for cand in candidates:
            memory = memories.get(cand.memory_id)
            if memory is None or memory.chunk_id is None:
                continue
            score = chunk_scores.get(memory.chunk_id, 0.0)
            if score > 0:
                entity_match[memory.id] = score

        scored = score_candidates(
            candidates,
            memories,
            plan,
            entity_match,
            operative_weight=settings.operative_rank_weight,
        )
        # Normalizers score_candidates uses, recomputed for the decomposition.
        max_rrf = max((c.rrf for c in scored), default=0.0) or 1.0
        max_entity = max(entity_match.values(), default=0.0)
        channel_max: dict[str, float] = {}
        for cand in scored:
            for name in INFORMATIVE_CHANNELS:
                s = cand.channel_scores.get(name)
                if s is not None:
                    channel_max[name] = max(channel_max.get(name, 0.0), s)

        final_pre = {c.memory_id: c.final for c in scored}
        pre_rank = {c.memory_id: i + 1 for i, c in enumerate(scored)}
        reranked = pipeline._rerank(user_id, plan, scored, memories)  # noqa: SLF001

        entry_rank = {cand.memory_id: i + 1 for i, cand in enumerate(reranked)}
        session_order: list[str] = []
        head_of: dict[str, int] = {}
        for cand in reranked:
            memory = memories.get(cand.memory_id)
            sid = memory.session_id if memory else None
            if sid and sid not in head_of:
                session_order.append(sid)
                head_of[sid] = cand.memory_id

        wide_settings = replace(settings, evidence_max_sessions=0)
        items_wide = assemble(
            wide_settings, store, user_id, plan, list(reranked), memories, top_k=100
        )
        items_ship = assemble(
            settings, store, user_id, plan, list(reranked), memories, top_k=100
        )
        emitted_ids = {i.memory_id for i in items_ship}
        window_ids = {i.memory_id for i in items_ship[:k]}
        wide_ids = {i.memory_id for i in items_wide}
        pooled_ids = {c.memory_id for c in candidates}
        admitted_ids = {c.memory_id for c in scored}

        def detail(memory_id: int) -> dict | None:
            cand = next((c for c in reranked if c.memory_id == memory_id), None)
            if cand is None:
                return None
            memory = memories.get(memory_id)
            strength = 0.0
            for name in INFORMATIVE_CHANNELS:
                s = cand.channel_scores.get(name)
                top = channel_max.get(name, 0.0)
                if s is not None and top > 0:
                    strength = max(strength, s / top)
            entity_raw = entity_match.get(memory_id, 0.0)
            sid = memory.session_id if memory else None
            return {
                "memory_id": memory_id,
                "session_id": sid,
                "is_session_head": head_of.get(sid) == memory_id if sid else False,
                "session_head_rank": (
                    session_order.index(sid) + 1 if sid in session_order else None
                ),
                "kind": memory.structural_kind if memory else None,
                "rank": entry_rank.get(memory_id),
                "in_pool": memory_id in pooled_ids,
                "admitted": memory_id in admitted_ids,
                "in_wide_payload": memory_id in wide_ids,
                "emitted": memory_id in emitted_ids,
                "in_window": memory_id in window_ids,
                "is_gold": memory_id in gold,
                "gold_scope": scope,
                "rrf": round(cand.rrf, 6),
                "base": round(cand.rrf / max_rrf, 4),
                "coverage": sum(
                    1 for n in INFORMATIVE_CHANNELS if n in cand.channels
                ) / len(INFORMATIVE_CHANNELS),
                "strength": round(strength, 4),
                "entity_raw": round(entity_raw, 4),
                "entity_norm": round(
                    entity_raw / max_entity, 4) if max_entity else 0.0,
                "entity_signal": round(
                    (entity_raw / max_entity) ** 1.5, 4) if max_entity else 0.0,
                "lexical_rank": cand.channels.get("lexical"),
                "entity_rank": cand.channels.get("entity"),
                "dense_rank": cand.channels.get("dense"),
                "dense_sim": round(cand.channel_scores.get("dense", 0.0), 4),
                "final_pre_rerank": round(final_pre.get(memory_id, cand.final), 4),
                "operative_plain": round(_operative_term(memory.text), 4) if memory else 0.0,
                "final": round(cand.final, 4),
                "pre_rank": pre_rank.get(memory_id),
                "promoted": None,
            }

        # What to dump: the heads of the first dozen sessions (the competition
        # the shipped payload actually faces) + every gold entry that made it
        # into the pool for a relevant session, plus that session's head.
        wanted: list[int] = [head_of[s] for s in session_order[:12]]
        per_session_gold: dict[str, list[int]] = {}
        for cand in reranked:
            if cand.memory_id not in gold:
                continue
            memory = memories.get(cand.memory_id)
            if memory is None or cand.memory_id not in pooled_ids:
                continue
            per_session_gold.setdefault(memory.session_id, []).append(cand.memory_id)
        for sid, ids in per_session_gold.items():
            wanted.append(ids[0])  # best-ranked gold entry of that session
            wanted.append(head_of.get(sid, ids[0]))  # and the entry that ranks it

        dumped = {memory_id: detail(memory_id) for memory_id in dict.fromkeys(wanted)}
        dumped = {m: d for m, d in dumped.items() if d}

        sess_files = {
            m["session_id"]: set(m.get("files") or ()) for m in data["memories"]
        }
        q_files = set(query.get("files") or ())
        pairs = []
        for sid in query["relevant"]:
            ids = per_session_gold.get(sid, [])
            n_total = entries_per_session.get(user_id, {}).get(sid, 0)
            pairs.append(
                {
                    "session_id": sid,
                    "overlap": len(q_files & sess_files.get(sid, set())),
                    "n_entries": n_total,
                    "n_gold_corpus": len(gold_by_session.get(sid, ())),
                    "n_gold_pooled": len(ids),
                    "n_pooled": sum(
                        1 for c in candidates
                        if memories.get(c.memory_id)
                        and memories[c.memory_id].session_id == sid
                    ),
                    "session_rank": (
                        session_order.index(sid) + 1 if sid in session_order else None
                    ),
                    "gold_best_rank": min((entry_rank[i] for i in ids), default=None),
                    "gold_admitted": any(i in admitted_ids for i in ids),
                    "gold_wide": any(i in wide_ids for i in ids),
                    "gold_emitted": any(i in emitted_ids for i in ids),
                    "gold_in_window": any(i in window_ids for i in ids),
                    "head_is_gold": head_of.get(sid) in gold if sid in head_of else None,
                }
            )
        wide_sessions_emitted = {
            (memories[i.memory_id].session_id if i.memory_id in memories else None)
            for i in items_wide
        }
        out.append(
            {
                "query_id": query["query_id"],
                "repo": query["repo"],
                "gold_scope": scope,
                "n_gold_corpus": len(gold),
                "n_gold_pooled": sum(p["n_gold_pooled"] for p in pairs),
                "sessions_in_pool": len(session_order),
                "entries_wide": len(items_wide),
                "sessions_wide": len({s for s in wide_sessions_emitted if s}),
                "entries_ship": len(items_ship),
                "pairs": pairs,
                "detail": list(dumped.values()),
            }
        )
        if len(out) % 20 == 0:
            print(f"  {len(out)} queries done", file=sys.stderr)
    return out


def report(rows: list[dict]) -> dict:
    pairs = [p for r in rows for p in r["pairs"]]
    strong = [p for p in pairs if p["overlap"] >= 2]
    print()
    print(f"queries {len(rows)}   relevant-session pairs {len(pairs)} ({len(strong)} strong)")
    print(
        f"gold-file memories in the whole user corpus: mean "
        f"{sum(r['n_gold_corpus'] for r in rows)/len(rows):.0f}"
    )

    def rate(subset, key):
        n = sum(1 for p in subset if p["n_entries"]) or 1
        return sum(1 for p in subset if p[key]) / n

    with_gold = [p for p in pairs if p["n_gold_pooled"]]
    print()
    print("A. how much of a relevant session is actually about the task's files")
    print(
        f"  relevant sessions with >=1 gold entry IN THE CORPUS: "
        f"{sum(1 for p in pairs if p['n_gold_corpus'])}/{len(pairs)}"
        f" ({sum(1 for p in pairs if p['n_gold_corpus'])/len(pairs):.1%})"
    )
    print(
        f"  relevant sessions with >=1 gold entry IN THE POOL : "
        f"{len(with_gold)}/{len(pairs)} ({len(with_gold)/len(pairs):.1%})"
    )
    print(
        f"    strong (>=2 shared files): {sum(1 for p in strong if p['n_gold_pooled'])}"
        f"/{len(strong)} pooled, "
        f"{sum(1 for p in strong if p['n_gold_corpus'])}/{len(strong)} in corpus"
    )
    print(
        f"  mean entries per relevant session: "
        f"{sum(p['n_entries'] for p in pairs)/len(pairs):.0f}   pooled: "
        f"{sum(p['n_pooled'] for p in pairs)/len(pairs):.1f}   gold in corpus: "
        f"{sum(p['n_gold_corpus'] for p in pairs)/len(pairs):.1f}   gold pooled: "
        f"{sum(p['n_gold_pooled'] for p in pairs)/len(pairs):.1f}"
    )
    print()
    print("B. survival of the GOLD entries (the right memory, not any chunk)")
    for key, label in [
        ("gold_admitted", "admitted for scoring"),
        ("gold_wide", "in payload (no session budget)"),
        ("gold_emitted", "emitted"),
        ("gold_in_window", "in the k-slot window"),
    ]:
        sub = with_gold
        print(
            f"  {label:32s}{rate(sub, key):>7.1%} of sessions that have a pooled gold entry"
        )
    print(
        f"  the session head is a gold entry: {sum(1 for p in pairs if p['head_is_gold'])/len(pairs):.1%}"
        "  (head = the chunk that carries the session's score into ranking)"
    )

    print()
    print("C. what beats a gold entry: decomposition against the leading session head")
    losers = [
        p
        for r in rows
        for p in r["pairs"]
        if p["overlap"] >= 2 and p["n_gold_pooled"] and (p["session_rank"] or 99) > 2
    ]
    gaps, winner_terms, loser_terms = [], [], []
    gold_beats_head = 0
    beats_both_heads = 0
    for r in rows:
        heads = [d for d in r["detail"] if d["is_session_head"]]
        winner = min((d for d in heads if d["session_head_rank"]), key=lambda d: d["session_head_rank"], default=None)
        if winner is None:
            continue
        top2 = [d for d in heads if (d["session_head_rank"] or 99) <= 2]
        for p in r["pairs"]:
            if p["overlap"] < 2 or not p["n_gold_pooled"]:
                continue
            gold = [
                d
                for d in r["detail"]
                if d["is_gold"] and d["session_id"] == p["session_id"]
            ]
            if not gold:
                continue
            best_gold = min(gold, key=lambda d: d["rank"] or 10**9)
            gaps.append(winner["final"] - best_gold["final"])
            terms = ("base", "coverage", "strength", "entity_signal", "operative_plain", "final_pre_rerank", "final")
            winner_terms.append({key: winner[key] for key in terms} | {"rank": winner["pre_rank"]})
            loser_terms.append({key: best_gold[key] for key in terms} | {"rank": best_gold["pre_rank"]})
            if best_gold["final_pre_rerank"] > winner["final_pre_rerank"]:
                gold_beats_head += 1
            if all(best_gold["final_pre_rerank"] > d["final_pre_rerank"] for d in top2):
                beats_both_heads += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(
        f"  strong relevant sessions with a pooled gold entry: {len(gaps)}"
        f"   of which ranked beyond session 2: {len(losers)}"
    )
    if gaps:
        print(f"  mean leading-head final - gold final: {mean(gaps):+.3f}")
        terms = ("base", "coverage", "strength", "entity_signal", "operative_plain", "final_pre_rerank", "final")
        for term in terms:
            print(
                f"  {term:16s} head {mean([t[term] for t in winner_terms]):.3f}"
                f"   gold {mean([t[term] for t in loser_terms]):.3f}"
                f"   gap {mean([t[term] for t in winner_terms]) - mean([t[term] for t in loser_terms]):+.3f}"
            )
        print(
            f"  pre-rerank entry position: head {mean([t['rank'] for t in winner_terms]):.1f}"
            f"   gold {mean([t['rank'] for t in loser_terms]):.1f}"
        )
        print(
            f"  gold entry already above the leading head BEFORE reranking: "
            f"{gold_beats_head}/{len(gaps)} ({gold_beats_head/len(gaps):.1%});"
            f" above both emitted heads: {beats_both_heads}/{len(gaps)}"
            f" ({beats_both_heads/len(gaps):.1%})"
        )
        print(
            f"  and the cross-encoder pushed it further down: gap grows "
            f"{mean([t['final_pre_rerank'] for t in winner_terms]) - mean([t['final_pre_rerank'] for t in loser_terms]):+.3f}"
            f" pre-rerank -> {mean(gaps):+.3f} post-rerank"
        )

    return {
        "pairs": len(pairs),
        "gold_in_corpus_rate": sum(1 for p in pairs if p["n_gold_corpus"]) / len(pairs),
        "gold_pooled_rate": len(with_gold) / len(pairs),
        "strong_gold_pooled_rate": (
            sum(1 for p in strong if p["n_gold_pooled"]) / len(strong) if strong else None
        ),
        "strong_gold_corpus_rate": (
            sum(1 for p in strong if p["n_gold_corpus"]) / len(strong) if strong else None
        ),
        "gold_wide_rate": rate(with_gold, "gold_wide"),
        "gold_emitted_rate": rate(with_gold, "gold_emitted"),
        "head_is_gold_rate": sum(1 for p in pairs if p["head_is_gold"]) / len(pairs),
        "mean_entries_ship": sum(r["entries_ship"] for r in rows) / len(rows),
        "mean_entries_wide": sum(r["entries_wide"] for r in rows) / len(rows),
        "mean_sessions_in_pool": sum(r["sessions_in_pool"] for r in rows) / len(rows),
        "mean_entries": mean([p["n_entries"] for p in pairs]),
        "mean_pooled": mean([p["n_pooled"] for p in pairs]),
        "mean_gold_corpus": mean([p["n_gold_corpus"] for p in pairs]),
        "mean_gold_pooled": mean([p["n_gold_pooled"] for p in pairs]),
        "decomposition_n": len(gaps),
        "term_gaps": {
            term: mean([t[term] for t in winner_terms]) - mean([t[term] for t in loser_terms])
            for term in ("base", "coverage", "strength", "entity_signal", "operative_plain", "final_pre_rerank", "final")
        } if gaps else {},
        "gold_above_leading_head_pre_rerank": gold_beats_head,
        "gold_above_both_heads_pre_rerank": beats_both_heads,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("eval/data/benchmark.json"))
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None, help="extra pass at this recall_per_channel")
    ap.add_argument("--pool", type=int, default=None, help="candidate_pool for the extra pass")
    ap.add_argument(
        "--operative-weights",
        type=float,
        nargs="+",
        default=None,
        help="extra passes, one per value, with this operative_rank_weight",
    )
    ap.add_argument(
        "--position-weights",
        type=float,
        nargs="+",
        default=None,
        help="extra passes, one per value, with this evidence_position_weight "
             "(intra-session slot choice only)",
    )
    ap.add_argument(
        "--no-promotion",
        action="store_true",
        help="extra pass with evidence_operative_promotion=0, to price the "
             "existing intra-session lever against the position tilt",
    )
    ap.add_argument(
        "--promotion-4",
        action="store_true",
        help="extra pass with evidence_operative_promotion=4. Promotion currently "
             "fires only for the top 2 sessions, which is every session the "
             "shipped payload contains, so it already covers any slot the position "
             "tilt could win back; this pass isolates its contribution.",
    )
    ap.add_argument(
        "--pos-noprom",
        type=float,
        default=None,
        help="extra pass with this evidence_position_weight and "
             "evidence_operative_promotion=0, so the two intra-session levers can "
             "be told apart: promotion moves the session's FIRST operative chunk "
             "to slot 1, the tilt prefers the LAST one",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--dump-rows",
        type=Path,
        default=None,
        help="per-pass (query, session) rows, for a paired comparison between passes",
    )
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    data["queries"] = [q for q in data["queries"] if q["relevant"]]
    if args.limit:
        data["queries"] = data["queries"][: args.limit]

    settings = Settings(data_dir=Path(tempfile.mkdtemp()))
    container = Container(settings)
    passes: list[tuple[str, list[dict]]] = []
    try:
        build(container, data)
        passes.append(
            (
                f"baseline (promote={settings.evidence_operative_promotion}, "
                f"position=0, operative=0)",
                analyse(container, data, args.k),
            )
        )
        for weight in args.position_weights or []:
            settings.evidence_position_weight = weight
            passes.append(
                (f"evidence_position_weight={weight}", analyse(container, data, args.k))
            )
            settings.evidence_position_weight = 0.0
        if args.no_promotion:
            settings.evidence_operative_promotion = 0
            passes.append(("promotion off", analyse(container, data, args.k)))
            settings.evidence_operative_promotion = 2
        if args.promotion_4:
            settings.evidence_operative_promotion = 4
            passes.append(("promotion top-4", analyse(container, data, args.k)))
            settings.evidence_operative_promotion = 2
        if args.pos_noprom is not None:
            settings.evidence_position_weight = args.pos_noprom
            settings.evidence_operative_promotion = 0
            passes.append(
                (
                    f"position={args.pos_noprom} promotion off",
                    analyse(container, data, args.k),
                )
            )
            settings.evidence_position_weight = 0.0
            settings.evidence_operative_promotion = 2
        for weight in args.operative_weights or []:
            settings.operative_rank_weight = weight
            passes.append((f"operative_rank_weight={weight}", analyse(container, data, args.k)))
        if args.depth or args.pool:
            settings.recall_per_channel = args.depth or settings.recall_per_channel
            settings.candidate_pool = args.pool or settings.candidate_pool
            passes.append(
                (f"depth={settings.recall_per_channel} pool={settings.candidate_pool}", analyse(container, data, args.k))
            )
    finally:
        container.close()

    summaries = {}
    raw = {}
    for label, rows in passes:
        print()
        print("=" * 70)
        print(f"pass: {label}")
        print("=" * 70)
        summaries[label] = report(rows)
        if args.dump_rows:
            # Per-pair rows, kept so two passes can be compared question by
            # question. Headline rates cannot show a paired sign test: +8 sessions
            # out of 135 could be 8 gained / 0 lost or 20 / 12, and those are
            # different results.
            raw[label] = [
                pair | {"query_id": row["query_id"]}
                for row in rows
                for pair in row["pairs"]
            ]
    if args.out:
        args.out.write_text(json.dumps(summaries, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.dump_rows:
        args.dump_rows.write_text(json.dumps(raw, indent=1), encoding="utf-8")
        print(f"wrote {args.dump_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
