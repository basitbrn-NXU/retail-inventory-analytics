"""Read-only views of frozen Module 4 results and the exported model."""
from pathlib import Path
import hashlib
import base64
import io
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CATEGORICAL = ["item_id", "store_id", "dept_id", "state_id", "event_name_1", "event_name_2"]
NUMERIC = ["wday", "month", "snap", "lag_7", "lag_28", "rolling_mean_28", "historical_sell_price"]
FEATURES = CATEGORICAL + NUMERIC
LABELS = {
    "rolling_mean_28": "Recent 28-day sales level", "wday": "Day of the week",
    "item_id": "Product identity", "lag_7": "Sales seven days earlier",
    "lag_28": "Sales 28 days earlier", "snap": "SNAP calendar indicator",
    "store_id": "Store identity", "historical_sell_price": "Last historical selling price",
    "event_name_1": "Calendar event", "event_name_2": "Second calendar event",
    "state_id": "State identity", "month": "Month", "dept_id": "Department identity",
}


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def features(frame):
    x = frame[FEATURES].copy()
    for col in CATEGORICAL:
        x[col] = x[col].fillna("No_event").astype(str)
    if not np.isfinite(x[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite model inputs.")
    return x


def load_model():
    manifest = read_json("registry.json")
    path = DATA / "selected_validation_model.joblib"
    payload = path.read_bytes() if path.exists() else base64.b64decode(
        (DATA / "selected_validation_model.joblib.b64").read_text(encoding="ascii"), validate=True)
    if hashlib.sha256(payload).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Model hash differs from the verified Module 4 model.")
    model = joblib.load(io.BytesIO(payload))  # Verified export; base64 changes transport only.
    sample = features(pd.DataFrame(manifest["example_request"]["rows"]))
    with threadpool_limits(limits=2):
        prediction = max(float(model.predict(sample)[0]), 0)
    if not np.isclose(prediction, manifest["example_prediction"], rtol=1e-10, atol=1e-10):
        raise ValueError("Model does not reproduce the Module 4 API example.")
    return model, manifest


def load_test():
    pred = pd.read_csv(DATA / "test_predictions.csv", parse_dates=["date"], float_precision="round_trip")
    base = pd.read_csv(DATA / "baseline_predictions.csv", parse_dates=["date"], float_precision="round_trip")
    joined = pred.merge(base[["id", "date", "prediction", "units"]],
                        on=["id", "date"], suffixes=("", "_baseline"), validate="one_to_one")
    if len(joined) != 2800 or not np.array_equal(joined.units, joined.units_baseline):
        raise ValueError("Archived evaluation records do not align.")
    return joined


def metrics(actual, predicted):
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(y) == 0:
        raise ValueError("Select at least one evaluation record.")
    denominator = y.sum()
    return {"MAE": float(np.abs(p-y).mean()), "RMSE": float(np.sqrt(np.mean((p-y)**2))),
            "WAPE": float(np.abs(p-y).sum()/denominator) if denominator else np.nan,
            "Bias": float((p-y).sum()/denominator) if denominator else np.nan}


def group_review(frame, dimension):
    mf = MetricFrame(metrics={"MAE": mean_absolute_error,
                             "RMSE": lambda y,p: metrics(y,p)["RMSE"],
                             "WAPE": lambda y,p: metrics(y,p)["WAPE"],
                             "Bias": lambda y,p: metrics(y,p)["Bias"]},
                     y_true=frame.units, y_pred=frame.prediction, sensitive_features=frame[dimension])
    table = mf.by_group.copy()
    table["Series"] = frame.groupby(dimension).id.nunique()
    table["Rows"] = frame.groupby(dimension).size()
    table["Limited coverage"] = table["Series"] < (5 if dimension == "store_id" else 10)
    table["Bias review"] = table.Bias.abs() > .10
    return table.reset_index().rename(columns={dimension: "Group"})


def scenario_row(example, history_percent=0, price_percent=0):
    if not -20 <= history_percent <= 20 or not -10 <= price_percent <= 10:
        raise ValueError("Scenario exceeds the declared interface limits.")
    row = dict(example["inputs"])
    for col in ["lag_7", "lag_28", "rolling_mean_28"]:
        row[col] *= 1 + history_percent/100
    row["historical_sell_price"] *= 1 + price_percent/100
    return row


def make_api(model, manifest, audit_path=None):
    # The Module 4 create_app factory and request schemas are reused unchanged.
    from module4_api import create_app
    if audit_path is None:
        audit_path = Path(tempfile.gettempdir()) / "ban6800_module5" / "api_audit.jsonl"
    return create_app(model, manifest, Path(audit_path))


def api_predict(model, manifest, row):
    from fastapi.testclient import TestClient
    # Same /predict route and validation, dispatched within the app process.
    with threadpool_limits(limits=2), TestClient(make_api(model, manifest)) as client:
        response = client.post("/predict", json={"rows": [row]})
        response.raise_for_status()
        return response.json()
