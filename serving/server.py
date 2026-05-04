"""Minimal HTTP server for TimesFM 2.5 forecasting.

Request/response shape matches the Vertex AI Model Garden TimesFM endpoint so
clients can swap backends without changing payloads.
"""

import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import timesfm

MODEL_ID = os.environ.get("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
MAX_CONTEXT = int(os.environ.get("TIMESFM_MAX_CONTEXT", "1024"))
MAX_HORIZON = int(os.environ.get("TIMESFM_MAX_HORIZON", "256"))
PER_CORE_BATCH = int(os.environ.get("TIMESFM_PER_CORE_BATCH", "1"))


class Instance(BaseModel):
  input: list[float] = Field(..., min_length=1)
  horizon: int = Field(..., gt=0)
  return_quantiles: bool = True


class PredictRequest(BaseModel):
  instances: list[Instance] = Field(..., min_length=1)


class Prediction(BaseModel):
  point_forecast: list[float]
  quantile_forecast: list[list[float]] | None = None


class PredictResponse(BaseModel):
  predictions: list[Prediction]


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
  torch.set_float32_matmul_precision("high")
  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
  model.compile(
    timesfm.ForecastConfig(
      max_context=MAX_CONTEXT,
      max_horizon=MAX_HORIZON,
      per_core_batch_size=PER_CORE_BATCH,
      normalize_inputs=True,
      use_continuous_quantile_head=True,
      force_flip_invariance=True,
      infer_is_positive=True,
      fix_quantile_crossing=True,
    )
  )
  _state["model"] = model
  yield
  _state.clear()


app = FastAPI(lifespan=lifespan, title="timesfm-server")


@app.get("/health")
def health():
  return {"status": "ready" if "model" in _state else "loading"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
  if "model" not in _state:
    raise HTTPException(status_code=503, detail="model not ready")

  for inst in req.instances:
    if inst.horizon > MAX_HORIZON:
      raise HTTPException(
        status_code=400,
        detail=f"horizon {inst.horizon} exceeds max_horizon {MAX_HORIZON}",
      )

  batch_horizon = max(inst.horizon for inst in req.instances)
  inputs = [np.array(inst.input, dtype=np.float32) for inst in req.instances]
  point, quantiles = _state["model"].forecast(horizon=batch_horizon, inputs=inputs)

  return PredictResponse(
    predictions=[
      Prediction(
        point_forecast=point[i, : inst.horizon].tolist(),
        quantile_forecast=(
          quantiles[i, : inst.horizon].tolist() if inst.return_quantiles else None
        ),
      )
      for i, inst in enumerate(req.instances)
    ]
  )
