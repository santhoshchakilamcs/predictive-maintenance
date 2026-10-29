#!/usr/bin/env python3
"""Build the dashboard preview (dashboard/index.html) from the BI exports.

The HTML preview is for quick viewing. The same exports (engine_health,
fleet_summary, model_metrics, sensor_trends) import directly into Power BI
or Tableau for the production dashboard.
Run from the project root: python3 dashboard/build_dashboard.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "dashboard" / "exports"


def main() -> None:
    health = pd.read_csv(EXPORTS / "engine_health.csv")
    fleet = pd.read_csv(EXPORTS / "fleet_summary.csv")
    metrics = pd.read_csv(EXPORTS / "model_metrics.csv")
    trends = pd.read_csv(EXPORTS / "sensor_trends.csv")
    trends = trends.iloc[::5]  # thin for the preview chart
    m = dict(zip(metrics["metric"], metrics["value"]))
    rmse = m.get("test_rmse", float("nan"))
    mae = m.get("test_mae", float("nan"))

    fleet_data = [{"name": r["health_status"], "value": int(r["engines"])}
                  for _, r in fleet.iterrows()]
    scatter = [[round(float(r["true_rul"]), 1), round(float(r["pred_rul"]), 1)]
               for _, r in health.iterrows()]
    cycles = trends["cycle"].tolist()
    series = {}
    for col in ["sensor_7", "sensor_11", "sensor_13"]:
        if col in trends.columns:
            series[col] = [round(float(v), 2) for v in trends[col]]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Predictive Maintenance Dashboard (preview)</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}h1{{font-size:22px}}
.cards{{display:flex;gap:16px}}.card{{border:1px solid #ddd;border-radius:8px;padding:12px 20px}}
.card b{{font-size:24px;display:block}}.chart{{width:100%;height:340px;margin-top:24px}}</style>
</head><body>
<h1>Predictive Maintenance: Turbofan Fleet Health (preview)</h1>
<p>Model: LSTM RUL regression on NASA C-MAPSS FD001, tracked in MLflow.
Same data as the CSV exports, which import into Power BI or Tableau.</p>
<div class="cards">
<div class="card"><span>Test RMSE (cycles)</span><b>{rmse:.1f}</b></div>
<div class="card"><span>Test MAE (cycles)</span><b>{mae:.1f}</b></div>
<div class="card"><span>Test engines</span><b>{len(health)}</b></div>
<div class="card"><span>Critical engines</span><b>{int((health['health_status'] == 'CRITICAL').sum())}</b></div>
</div>
<div id="donut" class="chart"></div>
<div id="scatter" class="chart"></div>
<div id="trends" class="chart"></div>
<script>
var donut = echarts.init(document.getElementById('donut'));
donut.setOption({{title:{{text:'Fleet health status'}},tooltip:{{}},series:[{{type:'pie',radius:'55%',data:{json.dumps(fleet_data)}}}]}});
var scatter = echarts.init(document.getElementById('scatter'));
scatter.setOption({{title:{{text:'Predicted vs true RUL (test engines)'}},xAxis:{{name:'True RUL'}},yAxis:{{name:'Predicted RUL'}},series:[{{type:'scatter',symbolSize:7,data:{json.dumps(scatter)}}}]}});
var trends = echarts.init(document.getElementById('trends'));
trends.setOption({{title:{{text:'Fleet-mean sensor trends over cycles'}},tooltip:{{trigger:'axis'}},legend:{{data:{json.dumps(list(series.keys()))}}},xAxis:{{type:'category',data:{json.dumps(cycles)}}},yAxis:{{}},series:[{','.join("{name:'" + k + "',type:'line',data:" + json.dumps(v) + "}" for k, v in series.items())}]}});
</script></body></html>"""
    (ROOT / "dashboard" / "index.html").write_text(html)
    print(f"wrote dashboard/index.html (rmse={rmse:.2f}, mae={mae:.2f})")


if __name__ == "__main__":
    main()
