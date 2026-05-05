"""Run TimesFM forecasts on each showcase series and write per-series CSVs.

Reads serving/showcase/data/*.csv, posts each series' context to a running
timesfm-server at localhost:8080 (with return_backcast=true), and writes
serving/showcase/forecasts/{series_id}.csv with predictions aligned to the
original timestamps. Filenames in data/ and forecasts/ match by series_id.

Output columns per file:
    timestamp, split, actual,
    predicted_mean, predicted_q10, predicted_q50, predicted_q90,
    is_anomaly

For context rows, predicted_* is the model's BACKCAST (its in-context
reconstruction); is_anomaly = actual outside backcast q10..q90 (skipping
the first 32 backcast values, which are unreliable due to left-padding).
For holdout rows, predicted_* is the model's FORECAST; is_anomaly =
actual outside forecast q10..q90 (i.e., the forecast missed).
"""

import csv
import glob
import json
import os
import sys
import urllib.request

URL = "http://localhost:8080/predict"
DATA_DIR = "serving/showcase/data"
OUT_DIR = "serving/showcase/forecasts"

# Must match the server's compiled config. The first BACKCAST_UNRELIABLE values
# of the returned backcast are predicted from prefixes containing left-padding
# zeros, so they are not meaningful for anomaly detection.
BACKCAST_UNRELIABLE = 32

FIELDNAMES = [
  "timestamp",
  "split",
  "actual",
  "predicted_mean",
  "predicted_q10",
  "predicted_q50",
  "predicted_q90",
  "is_anomaly",
]


def post(req: dict) -> dict:
  data = json.dumps(req).encode()
  r = urllib.request.urlopen(
    urllib.request.Request(
      URL, data=data, headers={"content-type": "application/json"}
    ),
    timeout=180,
  )
  return json.loads(r.read())


def forecast_one(input_path: str, output_path: str) -> tuple[int, int, int, int]:
  rows = list(csv.DictReader(open(input_path)))
  rows.sort(key=lambda r: r["timestamp"])
  context = [(r["timestamp"], float(r["value"])) for r in rows if r["split"] == "context"]
  holdout = [(r["timestamp"], float(r["value"])) for r in rows if r["split"] == "holdout"]
  if not context or not holdout:
    raise RuntimeError(f"{input_path} missing context or holdout")

  resp = post(
    {
      "instances": [
        {
          "input": [v for _, v in context],
          "horizon": len(holdout),
          "return_quantiles": True,
          "return_backcast": True,
        }
      ]
    }
  )
  pred = resp["predictions"][0]
  backcast_q = pred["backcast_quantiles"]
  forecast_q = pred["quantile_forecast"]

  ctx_flagged = 0
  hold_flagged = 0

  out_rows: list[dict] = []
  n_unbacked = len(context) - len(backcast_q)
  for i, (ts, actual) in enumerate(context):
    if i < n_unbacked:
      out_rows.append(_row(ts, "context", actual))
      continue
    bc_idx = i - n_unbacked
    q = backcast_q[bc_idx]
    reliable = bc_idx >= BACKCAST_UNRELIABLE
    is_anom = reliable and (actual < q[1] or actual > q[9])
    if is_anom:
      ctx_flagged += 1
    out_rows.append(_row(ts, "context", actual, q, is_anom))

  for i, (ts, actual) in enumerate(holdout):
    q = forecast_q[i]
    missed = actual < q[1] or actual > q[9]
    if missed:
      hold_flagged += 1
    out_rows.append(_row(ts, "holdout", actual, q, missed))

  with open(output_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    w.writerows(out_rows)

  return len(context), len(holdout), ctx_flagged, hold_flagged


def _row(ts, split, actual, q=None, is_anom=False):
  if q is None:
    return {
      "timestamp": ts,
      "split": split,
      "actual": f"{actual:.6g}",
      "predicted_mean": "",
      "predicted_q10": "",
      "predicted_q50": "",
      "predicted_q90": "",
      "is_anomaly": "",
    }
  return {
    "timestamp": ts,
    "split": split,
    "actual": f"{actual:.6g}",
    "predicted_mean": f"{q[0]:.6g}",
    "predicted_q10": f"{q[1]:.6g}",
    "predicted_q50": f"{q[5]:.6g}",
    "predicted_q90": f"{q[9]:.6g}",
    "is_anomaly": 1 if is_anom else 0,
  }


def main():
  if not os.path.isdir(DATA_DIR):
    print(f"missing {DATA_DIR}/ — run gen_showcase.py first", file=sys.stderr)
    sys.exit(1)
  os.makedirs(OUT_DIR, exist_ok=True)

  inputs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
  if not inputs:
    print(f"no CSVs in {DATA_DIR}", file=sys.stderr)
    sys.exit(1)

  print(
    f"{'series_id':<24}{'ctx':>6}{'hor':>6}{'ctx_flag':>10}{'hor_flag':>10}",
    file=sys.stderr,
  )
  for in_path in inputs:
    series_id = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(OUT_DIR, f"{series_id}.csv")
    ctx_n, hold_n, ctx_f, hold_f = forecast_one(in_path, out_path)
    print(
      f"{series_id:<24}{ctx_n:>6d}{hold_n:>6d}{ctx_f:>10d}{hold_f:>10d}",
      file=sys.stderr,
    )

  print(f"\nwrote {len(inputs)} forecast files to {OUT_DIR}/", file=sys.stderr)


if __name__ == "__main__":
  main()
