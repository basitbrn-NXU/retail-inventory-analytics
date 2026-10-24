"""Train a review-only challenger when a genuinely later labelled window arrives."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ORIGINAL_TEST_END = pd.Timestamp("2016-05-22")
IDENTITY = ["item_id", "store_id", "dept_id", "state_id"]
REQUIRED = IDENTITY + ["id", "date", "units", "wday", "month", "snap", "event_name_1", "event_name_2", "historical_sell_price"]


def validate_window(data, start, cohort):
    start = pd.Timestamp(start)
    if start <= ORIGINAL_TEST_END:
        raise ValueError("Use a new window after 22 May 2016; the original held-out period is not a fresh evaluation.")
    missing = sorted(set(REQUIRED)-set(data.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    d = data.copy()
    d["date"] = pd.to_datetime(d.date, errors="raise")
    if d.date.dt.tz is not None or (d.date != d.date.dt.normalize()).any():
        raise ValueError("Use timezone-naive daily dates.")
    if d.duplicated(["id", "date"]).any() or not np.isfinite(d.units.astype(float)).all() or (d.units < 0).any():
        raise ValueError("Invalid targets or duplicate series/date keys.")
    actual = set(map(tuple, d[IDENTITY].drop_duplicates().to_numpy()))
    expected = {tuple(row[k] for k in IDENTITY) for row in cohort}
    if actual != expected or d.id.nunique() != len(expected):
        raise ValueError("New data must retain the declared cohort; a new cohort requires a separate review.")
    if (d.groupby("id")[IDENTITY].nunique() > 1).any().any():
        raise ValueError("A series identifier cannot change its product/store mapping.")
    dates = pd.date_range(start, periods=28)
    window = d.loc[d.date.isin(dates)].sort_values(["date", "id"])
    history = d.loc[d.date < start].sort_values(["date", "id"])
    if len(window) != 28*len(expected) or set(window.date) != set(dates) or set(window.id) != set(history.id) or not window.groupby("id").size().eq(28).all():
        raise ValueError("The labelled evaluation window must cover every cohort member on 28 consecutive dates.")
    if history.date.nunique() < 84 or history.date.max() != start-pd.Timedelta(days=1):
        raise ValueError("Require at least 84 prior days ending immediately before the evaluation window.")
    for sid in history.id.unique():
        h = history.loc[history.id.eq(sid)].tail(28)
        if not h.date.reset_index(drop=True).equals(pd.Series(pd.date_range(start-pd.Timedelta(days=28), periods=28))):
            raise ValueError("Each cohort member needs 28 complete recent history days.")
    if not window.wday.between(1,7).all() or not window.month.between(1,12).all() or not window.snap.isin([0,1]).all():
        raise ValueError("Invalid known calendar inputs.")
    return history, window


def rebuild_features(history):
    h = history.sort_values(["id", "date"]).copy()
    g = h.groupby("id").units
    h["lag_7"] = g.shift(7)
    h["lag_28"] = g.shift(28)
    h["rolling_mean_28"] = g.transform(lambda s:s.shift(1).rolling(28, min_periods=28).mean())
    h["historical_sell_price"] = h.groupby("id").historical_sell_price.ffill()
    return h.dropna(subset=["lag_7", "lag_28", "rolling_mean_28", "historical_sell_price"])


def group_bias(frame):
    result = {}
    for dim in ["store_id", "dept_id"]:
        for name, f in frame.groupby(dim):
            result[f"{dim}={name}"] = float((f.prediction-f.units).sum()/f.units.sum()) if f.units.sum() else None
    return result


def gates(scores, champion_groups, challenger_groups):
    baseline = scores["baseline"].get("wape")
    candidate = scores["challenger"].get("wape")
    champion = scores["champion"].get("wape")
    comparable = baseline is not None and baseline > 0 and candidate is not None and champion is not None
    return {
        "baseline_relative_wape_gain_at_least_10pct": bool(comparable and 1-candidate/baseline >= .10),
        "not_worse_than_champion_wape": bool(comparable and candidate <= champion),
        "absolute_aggregate_bias_within_10pct": abs(scores["challenger"].get("aggregate_bias_fraction",float("inf"))) <= .10,
        "no_group_absolute_bias_deterioration_above_2points": all(
            c is not None and champion_groups.get(k) is not None and abs(c) <= abs(champion_groups[k])+.02
            for k,c in challenger_groups.items()) and bool(challenger_groups) and set(champion_groups)==set(challenger_groups),
    }


def run(data_path, window_start, output, tracking_uri):
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits
    import joblib
    import mlflow
    import mlflow.sklearn
    from module4.train_models import forecast, features, score
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"module5"))
    from core import load_model
    champion, manifest = load_model()
    data = pd.read_parquet(data_path)
    history, window = validate_window(data,window_start,manifest["cohort_combinations"])
    train = rebuild_features(history)
    with threadpool_limits(limits=2):
        challenger = clone(champion).fit(features(train),train.units)
        predictions = {name:forecast(model,history,window).merge(window[["id","date","units"]],on=["id","date"],validate="one_to_one")
                       for name,model in [("champion",champion),("challenger",challenger),("baseline",None)]}
    scores = {name:score(f.units,f.prediction) for name,f in predictions.items()}
    review_gates = gates(scores,group_bias(predictions["champion"]),group_bias(predictions["challenger"]))
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    model_path=output/"challenger.joblib";joblib.dump(challenger,model_path)
    record={"window_start":str(pd.Timestamp(window_start).date()),"history_end":str(history.date.max().date()),
            "source_sha256":hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),"champion_sha256":manifest["model_sha256"],
            "challenger_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),"scores":scores,"gates":review_gates,
            "technical_gates_passed":all(review_gates.values()),"production_approval":False,
            "promotion":"Named human review required; frozen serving model and candidate alias unchanged.",
            "group_bias":{name:group_bias(predictions[name]) for name in ["champion","challenger"]}}
    mlflow.set_tracking_uri(tracking_uri);mlflow.set_experiment("BAN6800_New_Window_Challengers")
    with mlflow.start_run(run_name="new_window_candidate") as active:
        record["mlflow_run_id"]=active.info.run_id
        mlflow.log_params({"window_start":record["window_start"],"history_end":record["history_end"],"data_hash":record["source_sha256"],"retuning":False})
        for name,metrics in scores.items():mlflow.log_metrics({f"{name}_{k}":float(v) for k,v in metrics.items()})
        mlflow.log_dict(record,"candidate_review.json")
        mlflow.sklearn.log_model(challenger,name="challenger")
        mlflow.set_tags({"approval":"pending_human_review","production_approval":"false"})
    (output/"candidate_review.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True);p.add_argument("--window-start",required=True)
    p.add_argument("--output",default="retraining/local-results")
    p.add_argument("--tracking-uri",default="sqlite:///tracking/new_window_candidates.db")
    a=p.parse_args();print(json.dumps(run(a.data,a.window_start,a.output,a.tracking_uri),indent=2))
