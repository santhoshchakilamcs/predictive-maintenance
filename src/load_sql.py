#!/usr/bin/env python3
"""Load NASA C-MAPSS FD001 into SQLite (the SQL warehouse layer).

Reads data/cmapss/train_FD001.txt, test_FD001.txt, RUL_FD001.txt, applies
schema.sql, and loads sensor_readings, rul_labels (train only, RUL capped
at 125 cycles in features.py, raw RUL stored here), and test_true_rul.
Run from the project root: python3 src/load_sql.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "predictive_maintenance.db"
COLS = (["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)])


def read_fd001(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLS
    return df


def main() -> None:
    train = read_fd001(ROOT / "data" / "cmapss" / "train_FD001.txt")
    test = read_fd001(ROOT / "data" / "cmapss" / "test_FD001.txt")
    rul = pd.read_csv(ROOT / "data" / "cmapss" / "RUL_FD001.txt",
                      sep=r"\s+", header=None, names=["true_rul"])
    rul["unit"] = range(1, len(rul) + 1)

    train["split"] = "train"
    test["split"] = "test"
    readings = pd.concat([train, test], ignore_index=True)

    # Train RUL: cycles remaining until each unit's final cycle.
    max_cycle = train.groupby("unit")["cycle"].max().rename("max_cycle")
    train_rul = train.join(max_cycle, on="unit")
    train_rul["rul"] = train_rul["max_cycle"] - train_rul["cycle"]
    labels = train_rul[["unit", "cycle", "rul"]].copy()
    labels["split"] = "train"

    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    try:
        con.executescript((ROOT / "sql" / "schema.sql").read_text())
        readings.to_sql("sensor_readings", con, if_exists="append", index=False)
        labels.to_sql("rul_labels", con, if_exists="append", index=False)
        rul[["unit", "true_rul"]].to_sql("test_true_rul", con,
                                         if_exists="append", index=False)
        con.commit()
        n_train = (readings["split"] == "train").sum()
        n_test = (readings["split"] == "test").sum()
        print(f"loaded readings: train={n_train} test={n_test} "
              f"units train={train['unit'].nunique()} test={test['unit'].nunique()}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
