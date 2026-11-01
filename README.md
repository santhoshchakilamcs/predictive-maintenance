# Predictive Maintenance: Turbofan RUL + Maintenance Copilot

End-to-end project covering **Python, SQL, Tableau/Power BI, deep learning,
generative AI (RAG), and MLflow**: predict turbofan engine remaining useful
life (RUL) from sensor streams, explain each alert with a retrieval copilot,
and serve the results on a fleet health dashboard.

## Architecture

```text
NASA C-MAPSS FD001 (train/test/RUL txt)
  -> src/load_sql.py          : SQL warehouse (SQLite: sensor_readings, rul_labels, predictions)
  -> src/features.py          : RUL capped at 125, 30-cycle sliding windows, MinMax scaling
  -> src/train.py             : PyTorch LSTM -> RUL regression, tracked in MLflow (params,
                                train/val/test RMSE/MAE, registered model artifact)
  -> src/predict.py           : test-engine RUL predictions + health status
                                (CRITICAL < 30 cycles, WARNING < 60, else HEALTHY)
  -> rag/ask.py               : TF-IDF retrieval over sample maintenance guides
                                (RAG copilot: grounded answers, no LLM call by design)
  -> dashboard/build_dashboard.py : BI exports (CSV, Power BI/Tableau-ready) + HTML preview
  -> app.py                   : FastAPI serving layer (/fleet, /predict/{unit}, /history/{unit}, /ask)
```

## Data provenance

Real NASA C-MAPSS Turbofan Engine Degradation Simulation data, FD001 subset
(100 train engines run to failure, 100 test engines, 1 operating condition,
1 fault mode: HPC degradation). Downloaded from the PHM data S3 mirror of the
NASA Prognostics Data Repository:
`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
Files live in `data/cmapss/` (train_FD001.txt, test_FD001.txt, RUL_FD001.txt).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # torch CPU, scikit-learn, mlflow, fastapi, ...
.venv/bin/python run_pipeline.py            # load -> train -> predict -> dashboard -> RAG demo
.venv/bin/mlflow ui --backend-store-uri ./mlruns
.venv/bin/uvicorn app:app --port 8000
```

Then open `dashboard/index.html`, or import `dashboard/exports/*.csv`
(engine_health, fleet_summary, model_metrics, sensor_trends) into Power BI
or Tableau.

## Per-engine alert history
`GET /history/{unit}` returns that engine's prediction, its alert
(CRITICAL if predicted RUL < 30, WARNING if < 60, else HEALTHY),
and its last 20 sensor cycles from `sensor_readings`, so a trend
can be plotted. Verified: unit 34 is CRITICAL, predicted RUL 4.96
at cycle 203, with trend cycles 184-203. Honest limit: the
predictions table has one row per test engine, so there is no
historical alert log in this demo, and this endpoint does not
invent past alerts. The RAG guides in `rag/docs/` are sample
documents, not OEM manuals, as noted below.

## Honest notes

- The RAG guides in `rag/docs/` are **sample documents written for this demo**,
  not OEM manuals. The copilot is retrieval-only (TF-IDF + cosine similarity),
  so every answer is grounded in retrieved text; plug an LLM generation step
  behind `rag/ask.py::ask()` when you want abstractive answers.
- SQLite is the demo warehouse (zero setup, like DataPilot's demo mode). Swap
  the connection string for Postgres/BigQuery for production.
- The dashboard preview is HTML/ECharts for quick viewing; the CSV exports are
  the production BI input, not a fake `.pbix`/`.twbx` file.
- Test RUL is evaluated on raw (uncapped) true RUL while training targets are
  capped at 125, which is standard for this benchmark; expect larger errors on
  healthy engines with high true RUL.
