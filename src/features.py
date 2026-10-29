#!/usr/bin/env python3
"""Shared feature engineering for the C-MAPSS FD001 RUL task.

- RUL is capped at 125 cycles (standard piecewise-linear target: early life
  is treated as healthy, degradation only matters near failure).
- Sensors with near-zero variance on the training set are dropped.
- Sliding windows of 30 cycles; the label is the capped RUL at the last
  cycle of each window. Test prediction uses each test unit's last window.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
COLS = (["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)])
RUL_CAP = 125
WINDOW = 30


def read_fd001(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLS
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    out = df.join(max_cycle, on="unit")
    out["rul"] = (out["max_cycle"] - out["cycle"]).clip(upper=RUL_CAP)
    return out


def select_features(train: pd.DataFrame) -> list:
    """Keep operating settings plus sensors that actually vary."""
    feature_cols = [c for c in COLS if c not in ("unit", "cycle")]
    std = train[feature_cols].std()
    kept = [c for c in feature_cols if std[c] > 1e-3]
    return kept


def make_windows(df: pd.DataFrame, feature_cols: list,
                 scaler: MinMaxScaler | None = None,
                 fit_scaler: bool = False):
    """Return (X, y, units): X shape (n_windows, WINDOW, n_features)."""
    if fit_scaler:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_cols])
    scaled = scaler.transform(df[feature_cols])
    df = df.copy()
    df[feature_cols] = scaled
    Xs, ys, units = [], [], []
    for _, group in df.sort_values(["unit", "cycle"]).groupby("unit"):
        vals = group[feature_cols].to_numpy(dtype=np.float32)
        ruls = group["rul"].to_numpy(dtype=np.float32)
        for i in range(len(group) - WINDOW + 1):
            Xs.append(vals[i:i + WINDOW])
            ys.append(ruls[i + WINDOW - 1])
            units.append(int(group["unit"].iloc[0]))
    return np.stack(Xs), np.array(ys), np.array(units), scaler


def last_windows(df: pd.DataFrame, feature_cols: list,
                 scaler: MinMaxScaler):
    """One window per unit (the last WINDOW cycles) for test prediction."""
    scaled = scaler.transform(df[feature_cols])
    df = df.copy()
    df[feature_cols] = scaled
    Xs, units, cycles = [], [], []
    for unit, group in df.sort_values(["unit", "cycle"]).groupby("unit"):
        vals = group[feature_cols].to_numpy(dtype=np.float32)
        Xs.append(vals[-WINDOW:])
        units.append(int(unit))
        cycles.append(int(group["cycle"].max()))
    return np.stack(Xs), np.array(units), np.array(cycles)
