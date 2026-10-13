"""Register the already trained model as an academic candidate, with evidence links."""
from __future__ import annotations
import argparse
import hashlib
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from module4.setup_tracking import ROOT,EXPERIMENT,inspect_data
from module4.train_models import features,TRAIN_END

NAME='BAN6800_M5_Household_Forecaster'


def register(parent):
    folder=ROOT/'models/module4'/parent
    selection=json.loads((folder/'selection.json').read_text())
    test=json.loads((folder/'test/summary.json').read_text())
    model_path=folder/'selected_validation_model.joblib'
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if test['model_sha256']!=digest or selection['parent_run_id']!=parent:raise ValueError('Model/evidence mismatch')
    reviews=list((folder/'review').glob('*/summary.json'))
    if not reviews:raise ValueError('Complete documentation before registration')
    review_path=max(reviews,key=lambda p:p.stat().st_mtime)
    review=json.loads(review_path.read_text())
    if review['model_sha256']!=digest:raise ValueError('Documentation uses a different model')
    source=ROOT/'data/processed/prepared.parquet';ready=inspect_data(source)
    data=pd.read_parquet(source);data['date']=pd.to_datetime(data.date)
    train=data[data.date<=TRAIN_END].sort_values(['date','id'])
    sample=features(train.head(5))
    local=joblib.load(model_path)
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'tracking/mlflow.db').as_posix());mlflow.set_experiment(EXPERIMENT)
    client=mlflow.MlflowClient()
    original=client.get_run(selection['source_run_id'])
    outputs=original.outputs.model_outputs if original.outputs else []
    models=[client.get_logged_model(entry.model_id) for entry in outputs]
    candidates=[model for model in models if model.name=='model']
    if len(candidates)!=1:raise ValueError('Expected exactly one logged model from the selected validation run')
    model_uri='models:/'+candidates[0].model_id
    logged=mlflow.sklearn.load_model(model_uri)
    if not np.allclose(logged.predict(sample),local.predict(sample),rtol=1e-10,atol=1e-10):
        raise ValueError('Logged model differs from the evaluated saved model')
    with mlflow.start_run(run_name='frozen_model_registration',tags={'phase':'registration','development_run_id':parent}) as run:
        versions=client.search_model_versions(f"name = '{NAME}'")
        matching=[v for v in versions if v.tags.get('model_sha256')==digest and v.tags.get('development_run_id')==parent]
        tags={'model_sha256':digest,'development_run_id':parent,'deployment_approved':'false',
              'lifecycle_status':'academic_candidate','test_run_id':test['test_run_id'],
              'documentation_run_id':review['review_run_id']}
        version=matching[0] if matching else mlflow.register_model(model_uri,NAME,tags=tags,await_registration_for=30)
        for key,value in tags.items():client.set_model_version_tag(NAME,version.version,key,value)
        client.set_registered_model_alias(NAME,'candidate',version.version)
        client.update_model_version(NAME,version.version,description='Frozen validation winner. Final test and XAI documented; substantial group-level errors remain. Academic local API demonstration only. Production deployment is not approved.')
        pairs=train[['item_id','store_id','dept_id','state_id']].drop_duplicates().to_dict('records')
        example=sample.iloc[0].to_dict()
        manifest={'registered_model_name':NAME,'version':str(version.version),'alias':'candidate',
                  'registry_uri':f'models:/{NAME}/{version.version}','logged_model_uri':model_uri,
                  'model_sha256':digest,'data_sha256':ready['sha256'],'development_run_id':parent,
                  'registration_run_id':run.info.run_id,'documentation_run_id':review['review_run_id'],
                  'lifecycle_status':'academic_candidate','deployment_approved':False,
                  'example_request':{'rows':[example]},'example_prediction':float(max(local.predict(sample.iloc[[0]])[0],0)),
                  'cohort_combinations':pairs,
                  'api_scope':'One-day prediction from supplied historical features; a full 28-day forecast requires the recursive forecasting routine.',
                  'stage_mapping':'MLflow aliases and lifecycle tags replace deprecated legacy stages. Candidate is for academic validation, not Production.'}
        (folder/'registry.json').write_text(json.dumps(manifest,indent=2))
        (folder/'api_request.json').write_text(json.dumps(manifest['example_request'],indent=2))
        mlflow.log_params({'registered_model_name':NAME,'version':str(version.version),'model_sha256':digest,'deployment_approved':False})
        mlflow.log_dict(manifest,'registry.json');mlflow.log_artifact(__file__,artifact_path='code')
        mlflow.log_artifact(str(review_path.parent/'model_card.md'));mlflow.log_artifact(str(review_path.parent/'fairness.md'))
        print('\nREGISTRATION COMPLETE')
        print('Model:',NAME,'| Version:',version.version,'| Alias: candidate')
        print('Saved manifest:',folder/'registry.json')
        print('Example API request:',folder/'api_request.json')
        print('Model unchanged. Lifecycle status: academic_candidate. Production approval: false.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run',required=True)
    args=parser.parse_args()
    if len(args.parent_run)!=32 or any(c not in '0123456789abcdef' for c in args.parent_run):parser.error('Invalid run ID')
    register(args.parent_run)


if __name__=='__main__':main()
