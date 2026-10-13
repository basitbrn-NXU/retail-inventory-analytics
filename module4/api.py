"""Local academic FastAPI demonstration for the fixed registered model version."""
from __future__ import annotations
import argparse
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,ConfigDict,Field
import uvicorn
from module4.setup_tracking import ROOT
from module4.train_models import features


class SalesRow(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    item_id:str=Field(min_length=1)
    store_id:str=Field(min_length=1)
    dept_id:str=Field(min_length=1)
    state_id:str=Field(min_length=1)
    event_name_1:str=Field(min_length=1)
    event_name_2:str=Field(min_length=1)
    wday:int=Field(ge=1,le=7,strict=True)
    month:int=Field(ge=1,le=12,strict=True)
    snap:int=Field(ge=0,le=1,strict=True)
    lag_7:float=Field(ge=0)
    lag_28:float=Field(ge=0)
    rolling_mean_28:float=Field(ge=0)
    historical_sell_price:float=Field(gt=0)


class PredictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    rows:list[SalesRow]=Field(min_length=1,max_length=100)


def create_app(model,manifest,audit_path):
    app=FastAPI(title='BAN6800 Retail Forecast API',version='1.0',
         description='Academic local demonstration of one-day feature-based predictions from a fixed registered model. Human review is required; production approval is false.')
    combinations={tuple(row[key] for key in ['item_id','store_id','dept_id','state_id']) for row in manifest['cohort_combinations']}
    categories=model.named_steps['encoding'].named_transformers_['category'].categories_
    event_sets=[set(categories[4]),set(categories[5])]
    def audit(record):
        audit_path.parent.mkdir(parents=True,exist_ok=True)
        with audit_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
    @app.get('/health')
    def health():return {'status':'ready','model':manifest['registered_model_name'],'version':manifest['version'],'deployment_approved':False}
    @app.post('/predict')
    def predict(request:PredictRequest):
        frame=pd.DataFrame([row.model_dump() for row in request.rows])
        for row in frame.itertuples(index=False):
            if (row.item_id,row.store_id,row.dept_id,row.state_id) not in combinations:
                raise HTTPException(status_code=422,detail='Product/store/department/state combination is outside the selected cohort.')
            if row.event_name_1 not in event_sets[0] or row.event_name_2 not in event_sets[1]:
                raise HTTPException(status_code=422,detail='Unseen event category; review inputs before prediction.')
        predictions=np.maximum(model.predict(features(frame)),0)
        if not np.isfinite(predictions).all():raise HTTPException(status_code=500,detail='Nonfinite model output')
        request_id=str(uuid.uuid4())
        audit({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'request_id':request_id,
               'model':manifest['registered_model_name'],'version':manifest['version'],
               'model_sha256':manifest['model_sha256'],'row_count':len(frame),'status':'success'})
        return {'request_id':request_id,'model':manifest['registered_model_name'],'version':manifest['version'],
                'predictions':predictions.tolist(),'unit':'daily sales units','deployment_approved':False}
    return app


def load_app(parent):
    folder=ROOT/'models/module4'/parent
    manifest=json.loads((folder/'registry.json').read_text())
    if manifest['development_run_id']!=parent:raise ValueError('Registry/run mismatch')
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'tracking/mlflow.db').as_posix())
    version=mlflow.MlflowClient().get_model_version(manifest['registered_model_name'],manifest['version'])
    if version.tags.get('model_sha256')!=manifest['model_sha256']:raise ValueError('Registered version hash tag mismatch')
    model=mlflow.sklearn.load_model(manifest['registry_uri'])
    sample=features(pd.DataFrame(manifest['example_request']['rows']))
    if not np.isclose(max(model.predict(sample)[0],0),manifest['example_prediction'],rtol=1e-10,atol=1e-10):
        raise ValueError('Registered model does not reproduce the verified example')
    return create_app(model,manifest,folder/'api/audit.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run',required=True)
    args=parser.parse_args()
    if len(args.parent_run)!=32 or any(c not in '0123456789abcdef' for c in args.parent_run):parser.error('Invalid run ID')
    app=load_app(args.parent_run)
    print('Open http://127.0.0.1:8000/docs to inspect the local /predict endpoint.',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=8000)


if __name__=='__main__':main()
