"""Poll a private Parquet file and train a review-only candidate on new labelled data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
from retraining.candidate import ORIGINAL_TEST_END, run


def check(data_path,state_path,output,tracking_uri):
    path=Path(data_path);state_path=Path(state_path)
    if not path.exists():return {"status":"waiting_for_private_dataset"}
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    previous=json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get("source_sha256")==digest:return {"status":"unchanged_private_dataset"}
    dates=pd.read_parquet(path,columns=["date"]).date
    start=pd.Timestamp(dates.max()).normalize()-pd.Timedelta(days=27)
    if start<=ORIGINAL_TEST_END:
        return {"status":"waiting_for_new_labelled_window","latest_date":str(pd.Timestamp(dates.max()).date())}
    result=run(path,start,output,tracking_uri)
    state_path.parent.mkdir(parents=True,exist_ok=True)
    state_path.write_text(json.dumps({"source_sha256":digest,"review_run_id":result["mlflow_run_id"]})+"\n")
    return {"status":"challenger_created_for_human_review","production_approval":False}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--state",default="retraining/local-results/watcher_state.json")
    p.add_argument("--output",default="retraining/local-results");p.add_argument("--tracking-uri",default="sqlite:///tracking/new_window_candidates.db")
    p.add_argument("--interval",type=int,default=3600);p.add_argument("--once",action="store_true");a=p.parse_args()
    while True:
        try:print(json.dumps(check(a.data,a.state,a.output,a.tracking_uri)),flush=True)
        except Exception as e:print(json.dumps({"status":"blocked_for_review","error":str(e)}),flush=True)
        if a.once:break
        time.sleep(max(60,a.interval))
