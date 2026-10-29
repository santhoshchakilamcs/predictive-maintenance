#!/usr/bin/env python3
"""FastAPI serving layer (precomputed predictions + RAG copilot).

Run from the project root: uvicorn app:app --port 8000
Endpoints: GET /health, GET /fleet, GET /predict/{unit}, GET /ask?q=...
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


@app.get("/ask")
def ask_endpoint(q: str):
    return ask(q)
