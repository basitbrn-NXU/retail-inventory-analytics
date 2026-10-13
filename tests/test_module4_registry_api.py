"""Check actual local model registration and validated prediction API calls."""
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from fastapi.testclient import TestClient
from test_module4_forecasting import fixture_data
from module4 import train_models as t
from module4 import register_model as reg
from module4.api import create_app


def fitted():
    data=fixture_data();train=data[data.split.eq('train')]
    model=t.build_model('random_forest',{'n_estimators':3,'max_depth':3,'min_samples_leaf':2})
    model.fit(t.features(train),train.units)
    return data,train,model


def test_registration_reuses_logged_model_and_version(tmp_path,monkeypatch):
    import mlflow.sklearn
    data,train,model=fitted();parent='a'*32
    folder=tmp_path/'models/module4'/parent;(folder/'test').mkdir(parents=True)
    (tmp_path/'tracking').mkdir()
    reg.mlflow.set_tracking_uri('sqlite:///'+(tmp_path/'tracking/mlflow.db').as_posix());reg.mlflow.set_experiment(reg.EXPERIMENT)
    with reg.mlflow.start_run(run_name='synthetic_selected') as run:
        source_run=run.info.run_id
        mlflow.sklearn.log_model(model,name='model',serialization_format='cloudpickle')
    joblib.dump(model,folder/'selected_validation_model.joblib')
    digest=hashlib.sha256((folder/'selected_validation_model.joblib').read_bytes()).hexdigest()
    (folder/'selection.json').write_text(json.dumps({'parent_run_id':parent,'source_run_id':source_run}))
    (folder/'test/summary.json').write_text(json.dumps({'model_sha256':digest,'test_run_id':'synthetic_test'}))
    review=folder/'review'/'fixture';review.mkdir(parents=True)
    (review/'summary.json').write_text(json.dumps({'model_sha256':digest,'review_run_id':'synthetic_review'}))
    (review/'model_card.md').write_text('Synthetic test card');(review/'fairness.md').write_text('Synthetic test review')
    source=tmp_path/'data/processed/prepared.parquet';source.parent.mkdir(parents=True);data.to_parquet(source)
    monkeypatch.setattr(reg,'ROOT',tmp_path);monkeypatch.setattr(reg,'inspect_data',lambda p:{'sha256':'synthetic_data'})
    reg.register(parent);reg.register(parent)
    manifest=json.loads((folder/'registry.json').read_text())
    assert manifest['version']=='1' and manifest['deployment_approved'] is False
    client=reg.mlflow.MlflowClient()
    assert len(client.search_model_versions(f"name = '{reg.NAME}'"))==1
    assert str(client.get_model_version_by_alias(reg.NAME,'candidate').version)=='1'
    from module4 import api
    monkeypatch.setattr(api,'ROOT',tmp_path)
    app_client=TestClient(api.load_app(parent))
    response=app_client.post('/predict',json=manifest['example_request'])
    assert response.status_code==200
    assert np.isclose(response.json()['predictions'][0],manifest['example_prediction'])


def test_api_prediction_validation_and_audit(tmp_path):
    _,train,model=fitted()
    example=t.features(train.iloc[[0]]).iloc[0].to_dict()
    manifest={'registered_model_name':'synthetic','version':'1','model_sha256':'synthetic_hash',
              'cohort_combinations':train[['item_id','store_id','dept_id','state_id']].drop_duplicates().to_dict('records')}
    audit=tmp_path/'audit.jsonl'
    client=TestClient(create_app(model,manifest,audit))
    response=client.post('/predict',json={'rows':[example]})
    assert response.status_code==200
    np.testing.assert_allclose(response.json()['predictions'],np.maximum(model.predict(t.features(pd.DataFrame([example]))),0))
    assert response.json()['deployment_approved'] is False
    assert len(audit.read_text().splitlines())==1
    assert 'lag_7' not in audit.read_text()
    assert client.post('/predict',json={'rows':[{**example,'item_id':'unknown'}]}).status_code==422
    assert client.post('/predict',json={'rows':[{**example,'lag_7':-1}]}).status_code==422
    assert client.post('/predict',json={'rows':[{**example,'event_name_1':'unseen'}]}).status_code==422
    assert client.post('/predict',json={'rows':[]}).status_code==422
