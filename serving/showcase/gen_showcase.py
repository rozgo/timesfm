"""Generate one CSV per showcase series for TimesFM demos.

Each series is procedurally generated to highlight a distinct model strength,
and each writes to its own CSV under serving/showcase/data/ so visualizations
can load them independently without conflating examples.

Run from repo root:
    python serving/showcase/gen_showcase.py

Outputs (one file per series):
    serving/showcase/data/web_traffic_hourly.csv
    serving/showcase/data/energy_load_hourly.csv
    ... etc.

Each CSV has columns: timestamp, value, split   (split ∈ {context, holdout})
"""

import csv
import os
from datetime import datetime, timedelta

import numpy as np

OUT_DIR = "serving/showcase/data"
SEED = 42
END_DATE = datetime(2026, 5, 4, 0, 0, 0)


def gen_timestamps(end: datetime, n: int, delta: timedelta) -> list[str]:
  """Return n ISO 8601 timestamps ending at `end` with given delta."""
  return [(end - delta * (n - 1 - i)).isoformat(sep=" ") for i in range(n)]


def web_traffic_hourly(rng: np.random.Generator):
  """Hourly visits with daily x weekly cycle and gentle upward trend.

  Showcases multi-period seasonality.
  """
  context, horizon = 1008, 168
  n = context + horizon
  t = np.arange(n)
  hour = t % 24
  day = (t // 24) % 7

  daily = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 8) / 24)
  weekly = np.where(day >= 5, 0.6, 1.0)
  trend = 1.0 + 0.01 * (t / (24 * 7))
  base = 200 + 800 * daily * weekly * trend
  values = base * rng.normal(1.0, 0.05, n)
  return values, context, horizon, timedelta(hours=1)


def energy_load_hourly(rng: np.random.Generator):
  """Hourly grid load: bimodal daily (8am, 6pm peaks), weekday boost, slow envelope.

  Showcases harmonic structure.
  """
  context, horizon = 1008, 168
  n = context + horizon
  t = np.arange(n)
  hour = t % 24
  day = (t // 24) % 7
  week = t // (24 * 7)

  morning = np.exp(-((hour - 8) ** 2) / 8)
  evening = np.exp(-((hour - 18) ** 2) / 8)
  daily = 400 + 400 * (morning + evening)
  weekday = np.where(day < 5, 1.0, 0.7)
  envelope = 1.0 + 0.15 * np.sin(2 * np.pi * week / 12)
  values = daily * weekday * envelope + rng.normal(0, 30, n)
  return values, context, horizon, timedelta(hours=1)


def temperature_daily(rng: np.random.Generator):
  """Daily mean temperature with yearly cycle, slight warming trend, weather AR(1).

  Showcases long-range periodic forecasting (yearly cycle from daily data).
  """
  context, horizon = 1024, 90
  n = context + horizon
  t = np.arange(n)
  year_frac = t / 365.25
  doy = (t % 365.25) / 365.25

  yearly = 12 + 10 * np.sin(2 * np.pi * (doy - 100 / 365.25))
  trend = 0.3 * year_frac
  weather = np.zeros(n)
  weather[0] = rng.normal(0, 2)
  for i in range(1, n):
    weather[i] = 0.7 * weather[i - 1] + rng.normal(0, 1.5)
  values = yearly + trend + weather
  return values, context, horizon, timedelta(days=1)


def saas_revenue_daily(rng: np.random.Generator):
  """Daily revenue: exponential growth, weekend dip, plateau periods around holidays.

  Showcases trend extrapolation alongside seasonality.
  """
  context, horizon = 912, 60
  n = context + horizon
  t = np.arange(n)
  dow = t % 7

  growth = 5000 * np.exp(0.0024 * t)
  weekly = np.where(dow >= 5, 0.6, 1.0)

  in_holiday = np.zeros(n, dtype=bool)
  for year_start in range(0, n, 365):
    for offset in (50, 150, 250, 350):
      d = year_start + offset
      if d < n:
        in_holiday[d : d + 5] = True

  values = growth * weekly + rng.normal(0, 200, n)
  for i in range(1, n):
    if in_holiday[i]:
      values[i] = values[i - 1] * 0.95 + rng.normal(0, 100)
  return np.maximum(values, 0), context, horizon, timedelta(days=1)


def server_cpu_5min(rng: np.random.Generator):
  """5-minute CPU% with smooth daily ramp + 3 planted anomalies in context.

  Showcases backcast anomaly detection — the headline use case for `return_backcast`.
  """
  context, horizon = 1008, 240
  n = context + horizon
  t = np.arange(n)
  hour = (t * 5 // 60) % 24

  daily = 25 + 12.5 * (np.sin(2 * np.pi * (hour - 14) / 24) + 1)
  noise = np.zeros(n)
  for i in range(1, n):
    noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 1.5)
  values = daily + noise

  # Planted anomalies (in context only, indices 0..1007):
  values[200] = 95.0  # single sharp spike
  values[500:512] += 30.0  # 1-hour sustained drift
  values[820] = 5.0  # midnight outlier (unusual low)

  return np.clip(values, 0, 100), context, horizon, timedelta(minutes=5)


def stock_returns_daily(rng: np.random.Generator):
  """Daily prices from Gaussian log-returns. No signal — model should show wide PIs.

  Showcases calibrated uncertainty when there's nothing to forecast.
  """
  context, horizon = 700, 30
  n = context + horizon
  returns = rng.normal(0, 0.015, n)
  prices = 100.0 * np.exp(np.cumsum(returns))
  return prices, context, horizon, timedelta(days=1)


def parts_demand_daily(rng: np.random.Generator):
  """Sparse intermittent demand: mostly zeros, occasional 1-5 unit orders.

  Showcases robust handling of non-Gaussian distributions.
  """
  context, horizon = 547, 30
  n = context + horizon
  t = np.arange(n)
  dow = t % 7
  doy = t % 365

  base = 0.3
  weekday = np.where(dow < 5, 1.0, 0.4)
  seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365)
  p = np.clip(base * weekday * seasonal, 0.05, 0.6)

  has_order = rng.uniform(0, 1, n) < p
  sizes = rng.choice([1, 1, 2, 2, 3, 4, 5], p=[0.35, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03], size=n)
  values = np.where(has_order, sizes, 0).astype(float)
  return values, context, horizon, timedelta(days=1)


GENERATORS = {
  "web_traffic_hourly": web_traffic_hourly,
  "energy_load_hourly": energy_load_hourly,
  "temperature_daily": temperature_daily,
  "saas_revenue_daily": saas_revenue_daily,
  "server_cpu_5min": server_cpu_5min,
  "stock_returns_daily": stock_returns_daily,
  "parts_demand_daily": parts_demand_daily,
}


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  rng = np.random.default_rng(SEED)

  summary = []
  for series_id, gen in GENERATORS.items():
    values, context_n, horizon_n, delta = gen(rng)
    ts = gen_timestamps(END_DATE, len(values), delta)
    out_path = os.path.join(OUT_DIR, f"{series_id}.csv")
    with open(out_path, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(["timestamp", "value", "split"])
      for i, (t, v) in enumerate(zip(ts, values)):
        split = "context" if i < context_n else "holdout"
        w.writerow([t, f"{v:.6g}", split])
    summary.append((series_id, context_n, horizon_n, float(values.min()), float(values.max())))

  print(f"wrote {len(GENERATORS)} files to {OUT_DIR}/")
  print(f"{'series_id':<24} {'context':>8} {'horizon':>8} {'min':>10} {'max':>10}")
  for s in summary:
    print(f"{s[0]:<24} {s[1]:>8d} {s[2]:>8d} {s[3]:>10.2f} {s[4]:>10.2f}")


if __name__ == "__main__":
  main()
