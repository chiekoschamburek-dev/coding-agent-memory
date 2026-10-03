"""Pure-RAG baseline: embed the query, cosine top-k over the same index.

Every stage that makes codemem a memory system rather than a vector search is
removed in one move: no query planning, no multi-channel recall, no identifier
scoring, no RRF fusion, no deterministic blend, no noise gate, no session
assembly, no cross-encoder. What remains is the canonical RAG shape — chunk
corpus, one query embedding, cosine similarity, top-k, chunk text verbatim.

The index is deliberately *not* rebuilt: this baseline adds memories through
the ordinary ``/add`` path and reads the vectors that Add already stored, so
the comparison against codemem isolates exactly the retrieval/assembly
machinery — same chunks, same embeddings, same benchmark, same judges.

``SearchService.handle`` is patched at runtime, so both harnesses
(``eval/run_benchmark.py``, ``eval/run_evidence.py``) run unchanged.

Usage::

    python scripts/baseline_dense_rag.py                # benchmark + evidence
    python scripts/baseline_dense_rag.py --mode rb      # benchmark only
    python scripts/baseline_dense_rag.py --mode ev      # evidence only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))


def dense_rag_handle(
    self, *, user_id: str, query: str, options: list[str] | None, top_k: int
) -> list:
    """The whole pure-RAG search: embed, cosine, take top-k, return verbatim.

    No similarity floor: a plain vector store returns whatever is nearest. No
    span selection, no dedup, no per-session quota: content is the stored chunk
    text whole, which is what an off-the-shelf RAG stack would feed the answer
    model.
    """
    from codemem.core.tokens import count_tokens
    from codemem.search.evidence import EvidenceItem, _iso_from_ms

    limit = max(1, min(top_k, self.settings.max_top_k))
    embedder = self.retriever.embedder
    if embedder is None or not embedder.available:
        return []
    vectors = embedder.embed([query])
    if not vectors:
        return []

    hits = self.store.dense_search(user_id, vectors[0], limit)
    if not hits:
        return []

    memory_ids = [memory_id for memory_id, _ in hits]
    memories = self.store.fetch_memories(user_id, memory_ids)
    chunk_ids = [
        memory.chunk_id
        for memory in memories.values()
        if memory.chunk_id is not None
    ]
    chunk_map = self.store.fetch_chunks(user_id, chunk_ids)

    items: list[EvidenceItem] = []
    for memory_id, score in hits:
        memory = memories.get(memory_id)
        if memory is None:
            continue
        chunk = chunk_map.get(memory.chunk_id) if memory.chunk_id else None
        text = chunk.text if chunk is not None else memory.text
        if not text:
            continue
        items.append(
            EvidenceItem(
                memory_id=memory.id,
                content=text,
                score=round(float(score), 6),
                created_at=_iso_from_ms(memory.ts) or memory.created_at,
                tokens=count_tokens(text),
                truncated=False,
                superseded=memory.superseded_by is not None,
            )
        )
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["both", "rb", "ev"], default="both")
    parser.add_argument("--data", type=Path, default=ROOT / "eval/data/benchmark.json")
    parser.add_argument("--qa", type=Path, default=ROOT / "eval/data/qa_modified.json")
    parser.add_argument("--out-rb", type=Path,
                        default=ROOT / "eval/results/rb_rag.json")
    parser.add_argument("--out-ev", type=Path,
                        default=ROOT / "eval/results/ev_rag.json")
    args = parser.parse_args(argv)

    from codemem.search.service import SearchPipeline

    SearchPipeline.handle = dense_rag_handle

    if args.mode in ("both", "rb"):
        import run_benchmark

        out_rb = args.out_rb
        print(f"=== pure-RAG proxy benchmark -> {out_rb}")
        run_benchmark.main([
            "--data", str(args.data),
            "--out", str(out_rb),
            "--dump-per-query", str(out_rb.with_name(out_rb.stem + "_pq.json")),
        ])

    if args.mode in ("both", "ev"):
        import run_evidence

        out_ev = args.out_ev
        print(f"=== pure-RAG evidence metric -> {out_ev}")
        run_evidence.main([
            "--qa", str(args.qa),
            "--data", str(args.data),
            "--json", str(out_ev),
            "--rows", str(out_ev.with_name(out_ev.stem + "_rows.json")),
            "--quiet",
        ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
