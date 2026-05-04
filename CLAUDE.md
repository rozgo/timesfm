# CLAUDE.md

Notes for future Claude sessions working in this fork.

## Repo orientation

This is a personal fork of `google-research/timesfm`.

- `origin` → `git@github.com:rozgo/timesfm.git` (this fork — push here)
- `upstream` → `git@github.com:google-research/timesfm.git` (read-only, pull from here to sync)

Existing structure (untouched):

- `src/timesfm/` — TimesFM 2.5 model code (PyTorch + Flax backends).
- `timesfm-forecasting/` — first-party Agent Skill (`SKILL.md`) plus examples.
- `v1/` — archived 1.0 / 2.0 model code and notebooks.
- `tests/` — unit tests for layers, configs, utilities.
- `AGENTS.md` — points agents at `timesfm-forecasting/SKILL.md`.

## What this fork adds: `serving/`

A minimal HTTP server wrapping `model.forecast()` so the model is reachable as a JSON API. Designed to be deployed eventually on Cloudflare Containers (see "Cloudflare port" below) but built and tested as a plain Docker image first.

### Files

- `serving/Dockerfile` — `python:3.11-slim`, CPU-only torch, installs timesfm from this repo's source, then a build-time warmup `RUN` that pre-fetches the 800 MB HuggingFace weights and pre-populates the `torch.compile` inductor cache.
- `serving/server.py` — FastAPI app with `GET /health` and `POST /predict`. Request/response shape matches the Vertex AI Model Garden TimesFM endpoint so clients can swap backends without changing payloads.

### API

```
POST /predict
{
  "instances": [
    {"input": [float, ...], "horizon": int,
     "return_quantiles": bool,    // default true
     "return_backcast": bool}      // default false
  ]
}
→
{
  "predictions": [
    {"point_forecast": [float ...horizon],
     "quantile_forecast": [[10 floats], ...horizon],
     "backcast": [float ...input_len],            // only when return_backcast=true
     "backcast_quantiles": [[10 floats], ...input_len]}
  ]
}
```

`quantile_forecast[t]` (and `backcast_quantiles[t]`) is `[mean, q10, q20, q30, q40, q50, q60, q70, q80, q90]`. `point_forecast` is the median.

### Backcast = retrospective view of the input

When `return_backcast=true`, the response includes the model's reconstruction of the input (its expected value at each input timestep, given the rest of the series). Use it for in-context anomaly detection:

```python
for t, actual in enumerate(input_values):
    q10, q90 = backcast_quantiles[t][1], backcast_quantiles[t][9]
    if actual < q10 or actual > q90:
        # anomaly
```

**Caveat**: TimesFM works in patches of 32. For inputs shorter than `max_context`, the first **~32 backcast values** are predictions made from prefixes that include left-padding zeros — they are **not reliable**. For anomaly detection on short series, skip `backcast[:32]`. For full-length inputs (≥ max_context), all returned backcast values are reliable.

The forecast portion of the response is unaffected by `return_backcast`. Setting it to true adds the backcast fields without changing `point_forecast` or `quantile_forecast`.

### Build / run / test

```bash
# from repo root (serving/Dockerfile uses repo-root context to install timesfm from src/)
docker build -f serving/Dockerfile -t timesfm-server .

docker run --rm -d --name timesfm-test -p 8080:8080 timesfm-server

curl -s localhost:8080/health
curl -s localhost:8080/predict -H 'content-type: application/json' \
  -d '{"instances":[{"input":[0,1,2,3,4,5,6,7,8,9],"horizon":12}]}' | jq
```

### Measured numbers (native arm64 on Apple Silicon)

| Metric | Value |
| --- | --- |
| Image size | 1.94 GB |
| Cold start (`docker run` → first 200 from `/predict`) | ~3 s |
| Steady-state latency (500-pt context, 24-step horizon, single instance) | ~630 ms median |
| Build time | ~6 min (mostly torch + weights download) |

### Why the build-time warmup matters

The `RUN python -c "..."` step at the end of the Dockerfile runs a dummy forecast. It writes:

1. The 800 MB model weights to the HuggingFace cache (`HF_HOME=/opt/hf-cache`).
2. The `torch.compile` inductor kernels to `/root/.cache/torch/inductor/`.

Both caches end up in the image layer. Without this, cold start would be ~30–40 s (weight download + JIT compile). With it, cold start is ~3 s (just Python import + state_dict load).

### IMPORTANT: shape config in warmup must match runtime

The warmup uses `max_context=1024, max_horizon=256, force_flip_invariance=True, ...`. The `server.py` runtime defaults match this exactly. **If you change either, change both** — otherwise the inductor cache misses and the first request post-cold-start triggers a fresh JIT compile (~10–20 s).

Env vars `TIMESFM_MAX_CONTEXT` / `TIMESFM_MAX_HORIZON` / `TIMESFM_PER_CORE_BATCH` override server.py defaults. They should also match the Dockerfile warmup or you give up the cache.

## Cloudflare port (next step, not done yet)

Plan agreed with the user but not implemented:

- Rebuild with `--platform=linux/amd64` (Cloudflare Containers is amd64-only — see <https://developers.cloudflare.com/containers/platform-details/architecture/>). The inductor cache is CPU-arch-specific; cross-arch builds need a fresh warmup pass on the target arch.
- Push image to a registry Cloudflare can pull from (Docker Hub, GHCR, R2-backed registry).
- Add a `wrangler.toml` with a Container binding to the image, instance type `standard-1` (1 vCPU / 4 GiB) or `standard-2` (2 vCPU / 8 GiB) for headroom.
- $5/month Workers Paid plan is required to deploy Containers.

The Vertex AI Model Garden TimesFM image (`us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/jax-timesfm-serve:...`) cannot be reused — it's GCP-private, expects Vertex's serving contract, and pulls weights from a `gs://vertex-model-garden-public-*` bucket at startup. We deliberately built our own image instead, which also gets us TimesFM **2.5** (the Model Garden image is still TimesFM **1.0** — `google/timesfm-v20240828`).

## Forecast quality flags (in `server.py` lifespan)

These are deliberately turned on:

- `normalize_inputs=True` — z-scores inputs before model, undoes after; prevents scale instability.
- `use_continuous_quantile_head=True` — uses the 1024-step quantile head for better PI calibration at long horizons.
- `force_flip_invariance=True` — guarantees `f(-x) = -f(x)` by running on `+x` and `-x` and averaging. **Doubles inference work.** Drop this for ~2× speedup if you don't need the symmetry guarantee.
- `infer_is_positive=True` — clamps forecast `≥ 0` if all inputs are `≥ 0`.
- `fix_quantile_crossing=True` — ensures `q10 ≤ q20 ≤ ... ≤ q90`.
- `return_backcast=True` — model emits its in-context reconstruction along with the forecast. Free (no extra deps) and required for retrospective anomaly detection. Per-request `return_backcast` field controls whether the backcast is *included in the JSON response*; this server-side flag controls whether it's *computed at all*.

See `src/timesfm/configs.py` for the full ForecastConfig and `src/timesfm/timesfm_2p5/timesfm_2p5_torch.py:352-487` for what each flag actually does in the compiled decode path.

## Things deliberately NOT built

- LLM-extractor agent layer (the "any doc → extract series → call /predict" path discussed). The server is the deterministic primitive; the LLM layer is a separate concern.
- Authentication on `/predict`. Add before exposing publicly.
- Rate limiting / concurrency control. The model is loaded once per process; concurrent requests share it via FastAPI's threadpool. For higher throughput, run multiple uvicorn workers or fronting Cloudflare Workers handle queueing.
- CSV upload endpoint. Trivial to add (parse → call `/predict` internally) — defer until needed.
