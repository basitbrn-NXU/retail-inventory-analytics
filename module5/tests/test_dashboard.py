from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from core import (DATA, api_predict, group_review, load_model, load_test,
                  make_api, metrics, read_json, scenario_row)

@pytest.fixture(scope="module")
def frozen():
    return load_model()

def test_assets_match_source_hashes():
    # Hashes were recorded on Windows (CRLF); Git stores text files with LF.
    # Accept either line ending so the check verifies content, not checkout settings.
    for name,digest in read_json("provenance.json")["files"].items():
        raw=(DATA/name).read_bytes()
        lf=raw.replace(b"\r\n",b"\n")
        candidates={hashlib.sha256(b).hexdigest() for b in (raw,lf,lf.replace(b"\n",b"\r\n"))}
        assert digest in candidates, name

def test_frozen_test_metrics_match_module4():
    frame=load_test();summary=read_json("test_summary.json")
    for col,stored in [("prediction","metrics"),("prediction_baseline","baseline_metrics")]:
        m=metrics(frame.units,frame[col])
        for ours,theirs in [("MAE","mae"),("RMSE","rmse"),("WAPE","wape"),("Bias","aggregate_bias_fraction")]:
            assert m[ours]==pytest.approx(summary[stored][theirs],abs=1e-12)
    assert frame.id.nunique()==100
    assert frame.date.nunique()==28

def test_fairlearn_group_metrics_match_recorded_review():
    frame=load_test();original=pd.read_csv(DATA/"original_group_review.csv")
    for dimension in ["store_id","dept_id"]:
        table=group_review(frame,dimension).set_index("Group")
        for row in original[original.dimension.eq(dimension)].itertuples():
            assert table.loc[row.group,"Bias"]==pytest.approx(row.aggregate_bias_fraction,abs=1e-12)
            assert table.loc[row.group,"RMSE"]==pytest.approx(row.rmse,abs=1e-12)
    assert bool(group_review(frame,"store_id").set_index("Group").loc["CA_4","Limited coverage"])

def test_examples_reproduce_predictions_and_shap(frozen):
    model,manifest=frozen
    examples=read_json("examples.json")
    assert len(examples)==3
    for e in examples:
        assert e["date"]=="2016-03-28"
        result=api_predict(model,manifest,scenario_row(e))
        assert result["predictions"][0]==pytest.approx(e["prediction"],abs=1e-10)
        assert e["base_value"]+sum(e["shap"].values())==pytest.approx(e["prediction"],abs=1e-10)
    assert scenario_row(examples[0],20,10)["lag_7"]==pytest.approx(examples[0]["inputs"]["lag_7"]*1.2)
    with pytest.raises(ValueError):scenario_row(examples[0],21,0)

def test_module4_api_validation_and_audit(frozen,tmp_path):
    model,manifest=frozen
    audit=tmp_path/"audit.jsonl"
    with TestClient(make_api(model,manifest,audit)) as client:
        assert client.get("/health").json()["deployment_approved"] is False
        response=client.post("/predict",json=manifest["example_request"])
        assert response.status_code==200
        assert response.json()["predictions"][0]==pytest.approx(manifest["example_prediction"])
        bad=json.loads(json.dumps(manifest["example_request"]))
        bad["rows"][0]["store_id"]="UNKNOWN"
        assert client.post("/predict",json=bad).status_code==422
        bad=json.loads(json.dumps(manifest["example_request"]))
        bad["rows"][0]["lag_7"]=-1
        assert client.post("/predict",json=bad).status_code==422
    record=json.loads(audit.read_text().splitlines()[0])
    assert record["row_count"]==1
    assert "inputs" not in record
    assert record["model_sha256"]==manifest["model_sha256"]

def test_all_dashboard_views_and_controls():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=30).run()
    assert not app.exception
    assert any("18.22%"==m.value for m in app.metric)
    app.multiselect(key="stores").set_value(["CA_4"]).run()
    assert not app.exception
    assert any("4 product" in c.value for c in app.caption)
    app.multiselect(key="stores").set_value([]).run()
    assert not app.exception and len(app.info)==1
    app.radio(key="page").set_value("Prediction examples").run()
    assert not app.exception and app.metric[0].value=="9.81 units"
    app.selectbox(key="example").set_value(2).run()
    assert not app.exception and app.metric[0].value=="6.70 units"
    app.radio(key="page").set_value("What-if explorer").run()
    assert not app.exception
    assert app.metric[0].value==app.metric[1].value
    app.slider(key="history").set_value(20).run()
    assert not app.exception
    assert app.metric[0].value!=app.metric[1].value
    app.radio(key="page").set_value("Fairness and ethics").run()
    assert not app.exception and len(app.dataframe)==2
    app.radio(key="dimension").set_value("Departments").run()
    assert not app.exception
    app.radio(key="page").set_value("Sources and scope").run()
    assert not app.exception
