#!/usr/bin/env python3
"""FastAPI serving layer (precomputed predictions + RAG copilot).

Run from the project root: uvicorn app:app --port 8000
Endpoints: GET /health, GET /fleet, GET /predict/{unit}, GET /history/{unit}, GET /ask?q=...
Predictions are precomputed by src/predict.py into SQLite; live model
inference for a new window is available via src/predict.py's model code.
"""
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rag.ask import ask  # noqa: E402

app = FastAPI(title="Predictive Maintenance API")
DB = ROOT / "data" / "predictive_maintenance.db"


def query(sql: str, params=()):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fleet")
def fleet():
    return query("SELECT health_status, COUNT(*) AS engines FROM predictions "
                 "GROUP BY health_status")


@app.get("/predict/{unit}")
def predict(unit: int):
    rows = query("SELECT * FROM predictions WHERE unit = ?", (unit,))
    if not rows:
        raise HTTPException(status_code=404, detail="unit not found")
    return rows[0]


@app.get("/history/{unit}")
def history(unit: int):
    """Per-engine alert history and sensor trend, from existing SQLite data.

    Honest limit: the predictions table has one row per test engine,
    so there is no historical alert log in this demo. This endpoint
    therefore returns the engine's current prediction as its alert
    (CRITICAL if predicted RUL < 30, WARNING if < 60, else HEALTHY,
    matching src/predict.py), plus the last 20 sensor cycles for
    that engine from sensor_readings, so a trend can be plotted.
    It does not invent past alerts."""
    rows = query("SELECT * FROM predictions WHERE unit = ?", (unit,))
    if not rows:
        raise HTTPException(status_code=404, detail="unit not found")
    pred = rows[0]
    trend = query(
        "SELECT cycle, sensor_2, sensor_7, sensor_11, sensor_15 "
        "FROM sensor_readings WHERE split = 'test' AND unit = ? "
        "ORDER BY cycle DESC LIMIT 20",
        (unit,))
    trend = list(reversed(trend))
    if pred["pred_rul"] < 30:
        alert = {"level": "CRITICAL", "cycle": pred["last_cycle"],
                 "message": "Predicted RUL under 30 cycles at last observed cycle."}
    elif pred["pred_rul"] < 60:
        alert = {"level": "WARNING", "cycle": pred["last_cycle"],
                 "message": "Predicted RUL under 60 cycles at last observed cycle."}
    else:
        alert = {"level": "HEALTHY", "cycle": pred["last_cycle"],
                 "message": "No alert at last observed cycle."}
    return {"prediction": pred, "alert": alert, "sensor_trend_last_20_cycles": trend}


@app.get("/ask")
def ask_endpoint(q: str):
    return ask(q)
