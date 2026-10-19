"""Module 4 request schemas and create_app factory, copied verbatim. See provenance.json."""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from core import features
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
