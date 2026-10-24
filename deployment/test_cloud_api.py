"""Integration checks for public API parity and honest monitoring boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from fastapi.testclient import TestClient
from deployment import cloud_api as api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Route audit metadata uses AUDIT_PATH set before importing the application.
    with api.lock:
        api.status_counts.clear()
        api.recent_bins.clear()
        api.group_counts.clear()
        api.total_prediction_rows = 0
        api.last_prediction_at = None
    with TestClient(api.app) as c:
        yield c


def test_frozen_model_contract_and_monitoring(client):
    response = client.post("/predict", json=api.manifest["example_request"])
    assert response.status_code == 200
    assert response.json()["predictions"][0] == pytest.approx(api.manifest["example_prediction"], abs=1e-10)
    assert response.json()["deployment_approved"] is False
    data = client.get("/monitoring/data").json()
    assert data["runtime"]["successful_prediction_rows"] == 1
    assert data["runtime"]["request_status_counts"] == {"200": 1}
    assert data["feature_drift"]["status"] == "insufficient_live_support"
    assert data["historical_evaluation"]["metrics"]["wape"] == pytest.approx(.545618075758467)
    assert "not_measurable" in data["live_group_error_and_bias"]["status"]
    assert data["model_sha256"] == api.manifest["model_sha256"]


def test_invalid_requests_are_counted_without_becoming_drift_inputs(client):
    bad = deepcopy(api.manifest["example_request"])
    bad["rows"][0]["lag_7"] = -1
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", content="invalid", headers={"Content-Type": "application/json"}).status_code == 422
    state = client.get("/monitoring/data").json()
    assert state["runtime"]["successful_prediction_rows"] == 0
    assert state["runtime"]["request_status_counts"] == {"422": 2}
    assert state["feature_drift"]["window_rows"] == 0


def test_live_drift_minimum_support_and_prometheus(client):
    rows = api.manifest["example_request"]["rows"] * 100
    assert client.post("/predict", json={"rows": rows}).status_code == 200
    state = client.get("/monitoring/data").json()
    assert state["feature_drift"]["window_rows"] == 100
    assert len(state["feature_drift"]["features"]) == 7
    assert all(v["psi"] >= 0 for v in state["feature_drift"]["features"].values())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ban6800_predict_requests_total" in response.text
    assert "ban6800_predict_seconds_count" in response.text


def test_monitoring_html_health_and_request_limit(client):
    assert client.get("/health").json()["deployment_approved"] is False
    page = client.get("/monitoring")
    assert page.status_code == 200
    assert "Historical accuracy reference" in page.text
    assert "No orders are placed" in page.text
    assert "HOUSEHOLD_2" in page.text
    assert client.post("/predict", content="x" * 128001).status_code == 413
    assert client.get("/monitoring/data").json()["runtime"]["request_status_counts"] == {"413": 1}


def test_reference_uses_training_only():
    assert api.reference["split"] == "train"
    assert api.reference["train_rows"] == 73000
    assert api.reference["reference_end"] == "2016-03-27"
    for item in api.reference["features"].values():
        assert sum(item["proportions"]) == pytest.approx(1)
        assert len(item["edges"]) + 1 == len(item["proportions"])
