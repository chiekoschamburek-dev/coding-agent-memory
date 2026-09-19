# Deployment

The platform requires a **self-hosted, publicly reachable** Add/Search API over
HTTP(S). It rejects targets with embedded credentials and those pointing at
private, loopback, or link-local addresses, and requires the endpoint to stay
available for at least 30 days.

GitHub is used for code hosting, CI, and image publication. It **cannot** host
the service: Pages serves static files only, Actions is a job runner rather than
a server, and Codespaces port forwarding is authenticated and sleeps when idle.

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

Pick one; switching later is cheap because only the reverse proxy changes.

### A. Small cloud VM + domain + automatic HTTPS (recommended)

A 2 vCPU / 4 GB instance and a cheap domain, roughly ¥100–200/year. The most
stable option and the easiest to keep up for 30 days. Caddy obtains and renews
certificates automatically.

`deploy/Caddyfile`:

```
your-domain.example {
    reverse_proxy codemem:8080
}
```

Then run the compose file plus Caddy on the same Docker network, and add
`CODEMEM_API_KEY`.

### B. Oracle Cloud Always Free VM

A genuinely free VM tier, at the cost of a more involved signup. Same procedure
as A once the instance exists.

### C. Hugging Face Spaces (Docker Space)

Free and fast to stand up (2 vCPU / 16 GB), but the Space sleeps when idle, so
keep-alive traffic is required. Suitable for smoke testing; less suitable as the
30-day endpoint unless kept warm.

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
