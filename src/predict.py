#!/usr/bin/env python3
"""Run inference on test engines, write predictions to SQLite + BI exports.

Run from the project root: python3 src/predict.py
Writes: predictions table in data/predictive_maintenance.db, plus
dashboard/exports/engine_health.csv, fleet_summary.csv, model_metrics.csv,
sensor_trends.csv (the last three are ready to import into Power BI or
Tableau). Health rule: pred RUL < 30 -> CRITICAL, < 60 -> WARNING, else
HEALTHY.
"""
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402
from train import RulLstm  # noqa: E402


def health_status(pred_rul: float) -> str:
    if pred_rul < 30:
        return "CRITICAL"
    if pred_rul < 60:
        return "WARNING"
    return "HEALTHY"


def main() -> None:
    models = ROOT / "models"
    sensor_cols = json.loads((models / "sensor_cols.json").read_text())
    scaler = joblib.load(models / "scaler.pkl")
    model = RulLstm(len(sensor_cols))
    model.load_state_dict(torch.load(models / "lstm_rul.pt", map_location="cpu"))
    model.eval()

    test = features.read_fd001(ROOT / "data" / "cmapss" / "test_FD001.txt")
    X_te, units, cycles = features.last_windows(test, sensor_cols, scaler)
    with torch.no_grad():
        preds = model(torch.tensor(X_te)).numpy()
    true = pd.read_csv(ROOT / "data" / "cmapss" / "RUL_FD001.txt",
                       sep=r"\s+", header=None, names=["true_rul"])["true_rul"].to_numpy()

    health = pd.DataFrame({
        "unit": units, "last_cycle": cycles, "true_rul": true, "pred_rul": preds,
    })
    health["abs_error"] = (health["true_rul"] - health["pred_rul"]).abs()
    health["health_status"] = health["pred_rul"].map(health_status)
    health["model_version"] = "lstm-fd001-v1"

    con = sqlite3.connect(ROOT / "data" / "predictive_maintenance.db")
    try:
        con.execute("DELETE FROM predictions")
        health.to_sql("predictions", con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()

    exports = ROOT / "dashboard" / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    health.to_csv(exports / "engine_health.csv", index=False)
    fleet = (health.groupby("health_status").size().rename("engines")
             .reset_index())
    fleet["share"] = fleet["engines"] / len(health)
    fleet.to_csv(exports / "fleet_summary.csv", index=False)
    rmse = float(np.sqrt(np.mean((health["true_rul"] - health["pred_rul"]) ** 2)))
    mae = float(health["abs_error"].mean())
    pd.DataFrame([{"metric": "test_rmse", "value": rmse},
                  {"metric": "test_mae", "value": mae},
                  {"metric": "n_test_engines", "value": len(health)}]
                 ).to_csv(exports / "model_metrics.csv", index=False)

    trend_sensors = [s for s in ["sensor_2", "sensor_4", "sensor_7", "sensor_8",
                                 "sensor_11", "sensor_13", "sensor_15", "sensor_20"]
                     if s in test.columns]
    trends = (test.groupby("cycle")[trend_sensors].mean().reset_index()
              .rename(columns={"cycle": "cycle"}))
    trends.to_csv(exports / "sensor_trends.csv", index=False)
    print(f"wrote {len(health)} predictions | test_rmse={rmse:.2f} test_mae={mae:.2f}")
    print(health["health_status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
