#!/usr/bin/env python3
"""Train the LSTM remaining-useful-life model, tracked in MLflow.

Run from the project root: python3 src/train.py
Artifacts: models/lstm_rul.pt, models/scaler.pkl, models/sensor_cols.json,
plus an MLflow run (local ./mlruns) with params, metrics, and the model.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402

import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402

EPOCHS = 100
HIDDEN = 64
BATCH = 256
LR = 1e-3


class RulLstm(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    train = features.add_rul(features.read_fd001(ROOT / "data" / "cmapss" / "train_FD001.txt"))
    test = features.read_fd001(ROOT / "data" / "cmapss" / "test_FD001.txt")
    sensor_cols = features.select_features(train)
    # Split by engine (not by window) so validation engines are truly unseen.
    train_units, val_units = train_test_split(train["unit"].unique(),
                                              test_size=0.15, random_state=42)
    X_tr, y_tr, _, scaler = features.make_windows(
        train[train["unit"].isin(train_units)], sensor_cols, fit_scaler=True)
    X_val, y_val, _, _ = features.make_windows(
        train[train["unit"].isin(val_units)], sensor_cols, scaler=scaler)
    train_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                          batch_size=BATCH, shuffle=True)
    val_X, val_y = torch.tensor(X_val), torch.tensor(y_val)

    model = RulLstm(len(sensor_cols))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    mlflow.set_experiment("predictive-maintenance-rul")
    with mlflow.start_run(run_name="lstm-fd001"):
        mlflow.log_params({"window": features.WINDOW, "rul_cap": features.RUL_CAP,
                           "hidden": HIDDEN, "epochs": EPOCHS, "lr": LR,
                           "n_features": len(sensor_cols),
                           "sensors": ",".join(sensor_cols)})
        for epoch in range(EPOCHS):
            model.train()
            tot = 0.0
            for xb, yb in train_dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            model.eval()
            with torch.no_grad():
                val_rmse = torch.sqrt(loss_fn(model(val_X), val_y)).item()
            train_rmse = (tot / len(X_tr)) ** 0.5
            mlflow.log_metrics({"train_rmse": train_rmse, "val_rmse": val_rmse},
                               step=epoch)
            print(f"epoch {epoch + 1}/{EPOCHS} train_rmse={train_rmse:.2f} "
                  f"val_rmse={val_rmse:.2f}")

        # Test evaluation: last window per test unit vs RUL_FD001.txt.
        X_te, units, _ = features.last_windows(test, sensor_cols, scaler)
        with torch.no_grad():
            preds = model(torch.tensor(X_te)).numpy()
        import pandas as pd
        true = pd.read_csv(ROOT / "data" / "cmapss" / "RUL_FD001.txt",
                           sep=r"\s+", header=None, names=["true_rul"])["true_rul"].to_numpy()
        rmse = float(np.sqrt(np.mean((preds - true) ** 2)))
        mae = float(np.mean(np.abs(preds - true)))
        mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae})
        try:
            mlflow.pytorch.log_model(model, "lstm_rul", input_example=X_te[:2])
        except Exception as e:
            print(f"warning: MLflow model logging failed ({e}); "
                  "local artifacts are still saved below")
        print(f"test_rmse={rmse:.2f} test_mae={mae:.2f} over {len(units)} test engines")

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models / "lstm_rul.pt")
    joblib.dump(scaler, models / "scaler.pkl")
    (models / "sensor_cols.json").write_text(json.dumps(sensor_cols))
    print(f"saved model + scaler + {len(sensor_cols)} sensor columns to {models}")


if __name__ == "__main__":
    main()
