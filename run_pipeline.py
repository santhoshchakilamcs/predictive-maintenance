#!/usr/bin/env python3
"""End-to-end pipeline runner: SQL load -> LSTM train (MLflow) -> predict
-> BI exports -> dashboard preview -> RAG demo query.

Run from the project root: python3 run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    print(f"\n=== {' '.join(args)} ===")
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    run("src/load_sql.py")
    run("src/train.py")
    run("src/predict.py")
    run("dashboard/build_dashboard.py")
    sys.path.insert(0, str(ROOT))
    from rag.ask import ask
    demo = ask("engine shows rising exhaust temperature and falling compressor efficiency")
    print("\n=== RAG demo ===")
    print(f"Q: {demo['query']}")
    for m in demo["matches"]:
        print(f"[{m['score']}] {m['title']}")
    print("\nDone. Next steps:")
    print("  mlflow ui --backend-store-uri ./mlruns   # experiment tracking")
    print("  uvicorn app:app --port 8000              # API")
    print("  open dashboard/index.html                # dashboard preview")
    print("  import dashboard/exports/*.csv into Power BI or Tableau")


if __name__ == "__main__":
    main()
