set -u
cd "/d/桌面/project/agent memory CSIG"
BM="eval/run_benchmark.py"; EV="eval/run_evidence.py"
BGE="BAAI/bge-reranker-v2-m3"
EVPIN="--cap 5 --max-sessions 2 --item-tokens 800 --position-weight 1.0 --operative-promotion 2 --prefix-tokens 1000 2000 4000 8000"

# Equal-pool attribution: everything matches arm M (pool 120) except the model
# and/or the window, so each contrast isolates exactly one factor.
echo "########## rb C1  bge pool=120 @2048/800   $(date +%H:%M:%S)"
PYTHONPATH=src python -u $BM --data eval/data/benchmark.json \
  --rerank-model $BGE --rerank-probability-scores --rerank-top-n 120 \
  --rerank-max-length 2048 --rerank-doc-tokens 800 \
  --out eval/results/rb_C1.json --dump-per-query eval/results/rb_C1_pq.json 2>&1 | tail -4

echo "########## rb C2  bge pool=120 @512/200    $(date +%H:%M:%S)"
PYTHONPATH=src python -u $BM --data eval/data/benchmark.json \
  --rerank-model $BGE --rerank-probability-scores --rerank-top-n 120 \
  --rerank-max-length 512 --rerank-doc-tokens 200 \
  --out eval/results/rb_C2.json --dump-per-query eval/results/rb_C2_pq.json 2>&1 | tail -4

echo "########## ev C1  $(date +%H:%M:%S)"
CODEMEM_RERANK_MODEL=$BGE CODEMEM_RERANK_PROBABILITY_SCORES=true \
CODEMEM_RERANK_TOP_N=120 CODEMEM_RERANK_MAX_LENGTH=2048 CODEMEM_RERANK_DOC_TOKENS=800 \
PYTHONPATH=src python -u $EV $EVPIN --json eval/results/ev_m_C1.json \
  --rows eval/results/ev_m_C1_rows.json --quiet 2>&1 | tail -3

echo "########## ev C2  $(date +%H:%M:%S)"
CODEMEM_RERANK_MODEL=$BGE CODEMEM_RERANK_PROBABILITY_SCORES=true \
CODEMEM_RERANK_TOP_N=120 CODEMEM_RERANK_MAX_LENGTH=512 CODEMEM_RERANK_DOC_TOKENS=200 \
PYTHONPATH=src python -u $EV $EVPIN --json eval/results/ev_m_C2.json \
  --rows eval/results/ev_m_C2_rows.json --quiet 2>&1 | tail -3

echo "########## ALL DONE $(date +%H:%M:%S)"
