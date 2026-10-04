# Deployment

The platform requires a **self-hosted, publicly reachable** Add/Search API over
HTTP(S). It rejects targets with embedded credentials and those pointing at
private, loopback, or link-local addresses, and requires the endpoint to stay
available for at least 30 days.

**As of 2026-09-26 every track must self-host.** The organisers withdrew the
"submit code, the platform builds it" path: the repository and the image are
disclosure material and do not substitute for a deployed endpoint. One of the
options below is therefore a submission prerequisite, not an option.

GitHub is used for code hosting, CI, and image publication. It **cannot** host
the service: Pages serves static files only, Actions is a job runner rather than
a server, and Codespaces port forwarding is authenticated and sleeps when idle.

## What cannot front this API

Add is synchronous. The internal guard is 1 500 s against the contract's
30-minute ceiling, and a large Add really can be a multi-minute connection that
returns nothing until it completes. So:

- **A CDN or edge proxy is unsafe here.** Alibaba Cloud ESA, for instance, caps
  the full-chain back-to-origin timeout at **300 s** (default 30 s, documented
  guidance "do not exceed 60 s") — far below a worst-case Add, so the platform
  would see a gateway error and record a failed submission. Its edge functions
  run JavaScript only, so it cannot host the container either.
- **A load balancer needs its idle timeout checked, not just its existence.**
  Four-layer LBs commonly idle out around 900 s, which also cuts a long Add.
- **Expose the container port directly on a VM.** `ECS:8080` behind a security
  group plus `CODEMEM_API_KEY` removes this whole failure class. If TLS or a
  domain is genuinely required, set the proxy read timeout above 1 800 s and
  confirm with a deliberately slow Add rather than assuming.

## Measured under load with the selection stage on (2026-10-04)

`scripts/loadtest.py`, 2 writers + 4 searchers, 45 s, select-llm enabled
(relay = gpt-4o-mini):

| scenario | search p50 | search p95 | errors | notes |
|---|---|---|---|---|
| relay healthy | **4.9 s** | 6.6 s | 0 | the selection call dominates latency: ~19 selections in the window, relay queuing under 4-way concurrency |
| relay broken (bad URL → instant fail) | 0.23 s | 0.35 s | 0 | the fallback is invisible to clients; throughput rises 20× |
| relay absent (no creds) | 0.29 s | 0.35 s | 0 | same |

Three operational facts for the one-shot window:

1. **Latency is a non-issue against the 30-minute per-request ceiling** —
   even the degraded-queueing case is 360× inside it. Do not tune this.
2. **Throughput collapses when selections are slow** (37 vs 775 searches in
   the window): if the platform issues searches concurrently, the relay's
   rate limit is the service's throughput limit. The fallback keeps every
   response contract-correct, so this degrades quality (no selection),
   never availability.
3. **Watch the relay balance for the whole 30-day window** — a drained
   balance fails every large prompt while tiny probes still pass (measured:
   `insufficient_user_quota`), which silently reverts the deployment to
   shipped-ordering quality. The `selection LLM call failed` warning count
   in the logs is the soak metric.

Also caught by this test and fixed in code: a CRLF `.env` puts a trailing
`` into `CODEMEM_LLM_BASE_URL`, which the OpenAI client rejects for every
selection call (measured: 388/775 silent fallbacks). `Settings` now strips
the URL/key/model; keep `.env` handling in mind if env is injected another
way.

## The image is host-agnostic

The same image runs anywhere, so the ingress choice does not affect the code:

```bash
docker build -t codemem:0.1.0 .
docker run -d --name codemem -p 8080:8080 \
  -v codemem-data:/data \
  -e CODEMEM_API_KEY=your-key \
  codemem:0.1.0
```

Verify:

```bash
curl -fsS http://127.0.0.1:8080/health
```

## Ingress options

Pick one. The service is one container with an embedded database, so moving
between these later means changing only where `docker run` happens.

### Sizing, measured on the proxy corpus

| need | evidence |
|---|---|
| Disk | 20 sessions produced a 16.6 MB database (1 904 memories) — about 250 MB per 300 sessions, 2.5 GB per 3 000. Vectors add ~1.5 KB per memory. The image itself is ~3–4 GB CPU, 10 GB+ GPU. A 40 GB system disk is ample. |
| RAM | Torch plus the two small encoders (bge-small 33M, MiniLM-L6 22M) holds roughly 1.5–2 GB resident; 8 GiB leaves room for a concurrent Add while searches run. |
| Add latency | Lexical path ~1.5 s per session; with dense enabled on CPU ~2.8 s per Add (300 Adds in ~850 s), against a 30-minute ceiling. |
| Search latency | 47 ms with the deterministic channels; ~1.9 s with the 120-document cross-encoder pool on CPU (15.9 ms/doc), ~0.67 s with dense+rerank on GPU. |

So **4 vCPU / 8 GiB is the comfortable floor for the full ranking**. A 2 vCPU /
4 GiB box is fine with `CODEMEM_DENSE_ENABLED=false` — measured on CPU, dense
added no metric gain for ~5x the Add cost, so turning it off there costs nothing.
1–2 GiB free tiers can only run the deterministic path
(`CODEMEM_RERANK_ENABLED=false` as well): MRR 0.7248 instead of 0.7718, which is
compliant but weaker.

### A. Small cloud VM, port exposed directly (recommended)

A 4 vCPU / 8 GiB economy instance, pay-as-you-go or annual (check the current
price on the vendor page). The most stable option and the easiest to keep up for
30 days: security group opens 8080, `CODEMEM_API_KEY` is set, and nothing sits
between the platform and uvicorn — which is what avoids the timeout class
described above.

A domain plus Caddy is optional, for TLS only. Caddy applies no response timeout
unless you configure one, so it does not cut a long Add; nginx by contrast
defaults `proxy_read_timeout` to 60 s and **will** kill it, so behind nginx set
that above 1 800 s and verify with a deliberately slow Add. `deploy/Caddyfile`:

```
your-domain.example {
    reverse_proxy codemem:8080
}
```

Then run the compose file plus Caddy on the same Docker network.

### B. Oracle Cloud Always Free VM

A genuinely free VM tier. **Note: on 2026-06-21 the Always Free Ampere A1
allowance was halved from 4 OCPU / 24 GB to 2 OCPU / 12 GB**, which still clears
the 8 GiB line above, so the full ranking runs there — but free-tier resources
can be reclaimed, so keep a snapshot and a migration path if this carries the
Full evaluation. Signup needs a card and is more involved.

### C. Hugging Face Spaces (Docker Space)

Free and fast to stand up (2 vCPU / 16 GB), but the Space sleeps when idle, so
keep-alive traffic is required. Suitable for smoke testing; less suitable as the
30-day endpoint unless kept warm. Also weigh the data obligations: request bodies
cross a third-party platform, and the rules forbid retaining or repurposing them.

### D. Tunnel from a local machine (zero cost)

Tailscale Funnel or Cloudflare Tunnel exposes a local container at a public
HTTPS URL. No server or domain needed for Funnel; Cloudflare needs a domain
(~¥10–70/year). Both require the machine and its network connection to stay up
for the full evaluation window, which is the risk to weigh.

## GPU

The default image is CPU-only so it runs anywhere. Dense retrieval and reranking
work on CPU but are roughly **8× slower** there, measured on the same corpus:

| stage | CPU | GPU (RTX 5060) |
|---|---|---|
| embedding, 256 long documents | 14 docs/s | 112 docs/s |
| reranking, 120-document pool | 15.9 ms/doc | 2.0 ms/doc |

That difference decides whether dense recall is worth its cost: on GPU it improves
nDCG@10 and recall@10 over rerank-only; on CPU it added no measurable gain. Both
are contract-compliant — 300 Add requests take ~155 s on GPU and ~850 s on CPU,
against a 30-minute per-request ceiling — but the ranking is better on GPU.

Build the CUDA image and pass the GPU through:

```bash
docker build -f deploy/Dockerfile.gpu -t codemem:0.1.0-gpu .
docker run -d --gpus all -p 8080:8080 -v codemem-data:/data \
  -e CODEMEM_API_KEY=your-key codemem:0.1.0-gpu
```

Requires the NVIDIA Container Toolkit on the host. Confirm it works:

```bash
docker run --rm --gpus all codemem:0.1.0-gpu \
  python3.12 -c "import torch; print(torch.cuda.is_available())"
```

`CODEMEM_EMBED_DEVICE` and `CODEMEM_RERANK_DEVICE` default to `auto`, which
selects CUDA when visible and falls back to CPU otherwise — so the GPU image is
also safe on a host where the GPU was not passed through, and the CPU image never
attempts to use one.

**Verify what actually loaded.** `GET /health` reports `channels`, including
whether the embedding and reranker models loaded. Without this, a silent fallback
to the CPU-only lexical path looks identical to success until the score is worse:

```json
{"status":"ok", ...,
 "channels":{"dense_enabled":true,"dense_available":true,
             "rerank_enabled":true,"rerank_available":true,"llm_enabled":false}}
```

## Operational notes

- **Persistence.** `CODEMEM_DATA_DIR` holds the SQLite database; always mount a
  volume or the corpus is lost on restart.
- **Auth.** Set `CODEMEM_API_KEY` for any real run. Leaving it unset is
  permitted for public smoke only.
- **Long Add calls.** Add is synchronous by contract and may legitimately take
  minutes. If a proxy sits in front, raise its request timeout; the platform
  allows up to 30 minutes.
- **Capacity.** Reads use a small connection pool; writes are serialized. A
  single container handles concurrent Search during an Add without blocking
  readers, courtesy of WAL mode.
- **Retries.** Add idempotency is keyed on `request_id`, so the platform's
  retries cannot duplicate memory.
- **Retention.** `DELETE /admin/users/{user_id}` erases a user's data; schedule
  it if required, and remember the 30-day deletion obligation.
