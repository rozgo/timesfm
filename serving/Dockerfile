# Build from the repo root:
#   docker build -f serving/Dockerfile -t timesfm-server .
#
# Run:
#   docker run --rm -p 8080:8080 timesfm-server
#
# Test:
#   curl -s localhost:8080/health
#   curl -s localhost:8080/predict -H 'content-type: application/json' \
#     -d '{"instances":[{"input":[0,1,2,3,4,5,6,7,8,9],"horizon":12}]}' | jq
#
# Note on architecture: this builds for the host's native arch. The torch.compile
# inductor cache baked at build time is CPU-arch specific. When porting to
# Cloudflare (linux/amd64 only), rebuild with: --platform=linux/amd64.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch first (skipping the CUDA wheels saves ~2 GB).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install timesfm from this repo's source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

# Server runtime deps.
RUN pip install fastapi 'uvicorn[standard]' pydantic numpy

# Server code.
COPY serving/server.py ./serving/server.py

# Warmup: pre-fetch HuggingFace weights AND pre-populate the torch.compile
# inductor cache. Both caches end up in this image layer, so the first
# request post-cold-start skips the JIT compile (~10-20s) and the weight
# download. The shape config here MUST match server.py defaults so the
# cached kernels are reused at runtime.
RUN python -c "\
import torch, numpy as np, timesfm; \
torch.set_float32_matmul_precision('high'); \
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch'); \
m.compile(timesfm.ForecastConfig( \
    max_context=1024, max_horizon=256, \
    normalize_inputs=True, use_continuous_quantile_head=True, \
    force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True, \
    return_backcast=True)); \
m.forecast(horizon=24, inputs=[np.zeros(1024, dtype=np.float32)]); \
print('[warmup] complete')"

WORKDIR /app/serving
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
