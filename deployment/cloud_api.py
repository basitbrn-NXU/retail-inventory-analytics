"""Public academic API and monitoring for the unchanged Module 4 model."""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CollectorRegistry, Counter as PromCounter, Histogram, generate_latest, CONTENT_TYPE_LATEST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module5"))
from core import load_model, make_api, read_json, group_review, load_test

THREADPOOL_LIMITER = threadpool_limits(limits=1)
model, manifest = load_model()
audit_path = Path(os.environ.get("AUDIT_PATH", str(Path(tempfile.gettempdir()) / "ban6800" / "api.jsonl")))
app = make_api(model, manifest, audit_path)
app.title = "BAN6800 Retail Forecast API and Monitoring"
app.description = "Academic demonstration of the frozen Module 4 model. Human review required. No stock orders or production approval."
registry = CollectorRegistry()
requests = PromCounter("ban6800_predict_requests", "Prediction requests by HTTP status", ["status"], registry=registry)
latency = Histogram("ban6800_predict_seconds", "Prediction route latency", registry=registry)
lock = Lock()
started_at = datetime.now(timezone.utc).isoformat()
status_counts = Counter()
recent_bins = deque(maxlen=1000)
group_counts = Counter()
total_prediction_rows = 0
last_prediction_at = None
reference_path = Path(__file__).with_name("reference.json")
reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
historical = read_json("test_summary.json")
historical_groups = {dim: group_review(load_test(), dim).to_dict(orient="records") for dim in ("store_id", "dept_id")}


def feature_bins(rows):
    if not reference:
        return []
    return [{name: int(np.searchsorted(spec["edges"], float(row[name]), side="right"))
             for name, spec in reference["features"].items()} for row in rows]


def drift_snapshot(window):
    if not reference:
        return {"status": "reference_not_configured", "features": {}}
    if len(window) < 100:
        return {"status": "insufficient_live_support", "minimum_rows": 100, "window_rows": len(window), "features": {}}
    result = {}
    for name, spec in reference["features"].items():
        counts = np.bincount([r[name] for r in window], minlength=len(spec["proportions"]))
        current = (counts + .5) / (counts.sum() + .5 * len(counts))
        expected = np.asarray(spec["proportions"], dtype=float)
        expected = np.maximum(expected, 1e-6)
        expected /= expected.sum()
        psi = float(np.sum((current - expected) * np.log(current / expected)))
        result[name] = {"psi": psi, "review": psi > .2}
    return {"status": "review" if any(v["review"] for v in result.values()) else "within_review_threshold",
            "window_rows": len(window), "window_limit": 1000, "psi_review_threshold": .2, "features": result}


@app.middleware("http")
async def instrument_prediction(request, call_next):
    global total_prediction_rows, last_prediction_at
    if request.url.path != "/predict" or request.method != "POST":
        return await call_next(request)
    # Read once through Starlette's cached request body. Never persist request features.
    body = await request.body()
    if len(body) > 128_000:
        code = 413
        with lock:
            status_counts[str(code)] += 1
        requests.labels(status=str(code)).inc()
        return Response("Request body exceeds 128 KB", status_code=code)
    start = perf_counter()
    code = 500
    try:
        response = await call_next(request)
        code = response.status_code
        if code == 200:
            rows = json.loads(body)["rows"]
            bins = feature_bins(rows)
            with lock:
                total_prediction_rows += len(rows)
                last_prediction_at = datetime.now(timezone.utc).isoformat()
                recent_bins.extend(bins)
                for row in rows:
                    group_counts[row["store_id"]] += 1
        return response
    finally:
        latency.observe(perf_counter() - start)
        requests.labels(status=str(code)).inc()
        with lock:
            status_counts[str(code)] += 1


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/data")
def monitoring_data():
    with lock:
        runtime = {"started_at_utc": started_at, "last_prediction_at_utc": last_prediction_at,
                   "request_status_counts": dict(status_counts), "successful_prediction_rows": total_prediction_rows,
                   "store_prediction_counts": dict(group_counts)}
        window = list(recent_bins)
    return {"model_sha256": manifest["model_sha256"], "model_version": manifest["version"],
            "deployment_approved": False, "runtime": runtime,
            "feature_drift": drift_snapshot(window),
            "historical_evaluation": {"period": "2016-04-25 to 2016-05-22", "rows": 2800,
                                      "metrics": historical["metrics"], "groups": historical_groups},
            "live_performance": {"status": "not_measurable_without_new_observed_sales_labels"},
            "live_group_error_and_bias": {"status": "not_measurable_without_new_observed_sales_labels"},
            "demographic_fairness": "not_assessable_no_protected_attributes_or_binary_decision",
            "retention": "Runtime counters and bin indices are held in memory; request audit metadata is temporary. Both reset on restart. No durable cloud monitoring history is claimed.",
            "drift_interpretation": "PSI measures input mix change, not accuracy loss or discrimination. User-entered scenarios can trigger drift. Review is advisory; no automatic retraining or model promotion."}


def html_table(rows):
    if not rows:
        return "<p>No observations yet.</p>"
    keys = list(rows[0])
    return "<table><thead><tr>" + "".join(f"<th>{escape(str(k))}</th>" for k in keys) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(k, '')))}</td>" for k in keys) + "</tr>" for row in rows) + "</tbody></table>"


@app.get("/monitoring", response_class=HTMLResponse, include_in_schema=False)
def monitoring_page():
    state = monitoring_data()
    runtime = state["runtime"]
    drift = state["feature_drift"]
    drift_labels = {"insufficient_live_support": "Waiting for 100 rows", "review": "Review input change",
                    "within_review_threshold": "Within review threshold", "reference_not_configured": "Reference unavailable"}
    last_prediction = runtime["last_prediction_at_utc"]
    last_label = datetime.fromisoformat(last_prediction).strftime("%d %b %Y %H:%M:%S") if last_prediction else "None since startup"
    cards = [("Successful prediction rows", runtime["successful_prediction_rows"]),
             ("Last prediction UTC", last_label),
             ("Input drift", drift_labels[drift["status"]])]
    tiles = "".join(f"<article><span>{escape(k)}</span><strong>{escape(str(v))}</strong></article>" for k, v in cards)
    status_rows = [{"HTTP status": k, "Requests": v} for k, v in runtime["request_status_counts"].items()]
    drift_rows = [{"Feature": k, "PSI": round(v["psi"], 4), "Review": v["review"]} for k, v in drift.get("features", {}).items()]
    drift_table = html_table(drift_rows) if drift_rows else f"<p>{drift.get('window_rows', 0)} accepted rows in the current window. Drift results appear after 100 rows.</p>"
    groups = []
    for dimension, values in historical_groups.items():
        for r in values:
            groups.append({"Dimension": dimension, "Group": r["Group"], "Series": r["Series"],
                           "MAE": round(r["MAE"], 3), "Bias percent": round(100*r["Bias"], 2),
                           "Review": r["Bias review"], "Limited coverage": r["Limited coverage"]})
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retail forecasting monitoring</title><style>
body{font-family:system-ui,sans-serif;color:#18304a;background:#f5f7fa;margin:0}main{max-width:1100px;margin:auto;padding:35px 24px}
h1{font-size:32px}h2{margin-top:34px}.cards{display:flex;gap:16px;flex-wrap:wrap}article{background:white;padding:20px;border-radius:9px;flex:1;min-width:190px}strong{display:block;margin-top:10px;font-size:21px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3ec}th{background:#e9eef5}.scroll{overflow:auto}.notice{padding:16px;background:#fff1d2;border-radius:8px}a{color:#00616d}p{line-height:1.6}code{overflow-wrap:anywhere}
</style><main><h1>Retail forecasting monitoring</h1><p>Academic demonstration · candidate version 1 · human review required</p>
<p><a href="/docs">API documentation</a> · <a href="/metrics">Prometheus metrics</a> · <a href="/monitoring/data">Monitoring JSON</a></p>
<div class="notice">Live input monitoring is separate from the historical evaluation below. New accuracy and group-bias outcomes cannot be measured until observed sales labels arrive.</div>
<h2>Live API activity since startup</h2><div class="cards">""" + tiles + "</div>" + html_table(status_rows) + \
        f"<p>Process started UTC: {escape(started_at)}. Refresh this page for current values.</p>" + \
        "<h2>Live input distribution</h2><p>Training-only reference. Minimum 100 accepted rows; rolling window up to 1,000 rows. PSI above 0.20 triggers review, not automatic retraining. Scenario queries may change the input mix.</p>" + drift_table + \
        "<h2>Historical accuracy reference</h2><p>Frozen final test, 25 April to 22 May 2016, 2,800 rows. MAE 2.3337 units; RMSE 3.4014 units; WAPE 54.56%; aggregate bias −1.37%. These are historical evaluation results.</p>" + \
        "<h2>Historical operational equity review</h2><p>Absolute group bias above 10% triggers review. CA_4 and HOUSEHOLD_2 have limited cohort support. Demographic parity, equalized odds and disparate impact are unassessed because customer demographics and a binary decision are absent.</p><div class='scroll'>" + html_table(groups) + "</div>" + \
        f"<h2>Provenance and retention</h2><p>Model SHA-256: <code>{manifest['model_sha256']}</code></p><p>{escape(state['retention'])}</p><p>No orders are placed. Production approval remains false.</p></main></html>"


@app.get("/", include_in_schema=False)
def index():
    return {"project": "BAN6800 Retail Forecasting", "api_docs": "/docs", "health": "/health",
            "monitoring": "/monitoring", "prometheus": "/metrics", "deployment_approved": False}
