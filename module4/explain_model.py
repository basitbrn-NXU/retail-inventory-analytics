"""Explain the frozen model on reconstructed validation forecast inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from module4.setup_tracking import ROOT, EXPERIMENT, inspect_data
from module4.train_models import TRAIN_END, FEATURES, CATEGORICAL, NUMERIC, KNOWN_FUTURE, features


def reconstruct_inputs(data, saved_predictions):
    history=data[data.date<=TRAIN_END].sort_values(['date','id'])
    ids=sorted(history.id.unique())
    values=history.pivot(index='date',columns='id',values='units').reindex(columns=ids).tail(28).to_numpy(dtype=float)
    prices=history.groupby('id')['historical_sell_price'].last().reindex(ids).to_numpy()
    future=data[data['split'].eq('validation')][KNOWN_FUTURE].copy()
    outputs=[]
    for day in sorted(future.date.unique()):
        frame=future[future.date.eq(day)].set_index('id').reindex(ids).reset_index()
        frame['lag_7']=values[-7];frame['lag_28']=values[-28]
        frame['rolling_mean_28']=values.mean(axis=0);frame['historical_sell_price']=prices
        outputs.append(frame)
        p=saved_predictions[saved_predictions.date.eq(day)].set_index('id').reindex(ids).prediction.to_numpy()
        if not np.isfinite(p).all():raise ValueError('Incomplete saved validation predictions')
        values=np.vstack([values[-27:],p])
    return pd.concat(outputs,ignore_index=True)


def grouped_shap(model, x):
    encoder=model.named_steps['encoding']
    encoded=encoder.transform(x)
    explainer=shap.TreeExplainer(model.named_steps['model'],feature_perturbation='tree_path_dependent',model_output='raw')
    explanation=explainer(encoded,check_additivity=True)
    contributions=np.asarray(explanation.values)
    grouped=[];position=0
    for categories in encoder.named_transformers_['category'].categories_:
        count=len(categories)
        grouped.append(contributions[:,position:position+count].sum(axis=1));position+=count
    for _ in NUMERIC:
        grouped.append(contributions[:,position]);position+=1
    if position!=contributions.shape[1]:raise ValueError('Unexpected encoded feature mapping')
    values=np.column_stack(grouped)
    base=np.asarray(explanation.base_values).reshape(-1)
    if len(base)==1:base=np.repeat(base,len(x))
    raw=model.predict(x)
    if not np.allclose(base+values.sum(axis=1),raw,rtol=1e-5,atol=1e-5):
        raise ValueError('Grouped SHAP contributions do not reconstruct raw predictions')
    return values,base,raw


def run_analysis(parent):
    folder=ROOT/'models/module4'/parent
    selection=json.loads((folder/'selection.json').read_text())
    if selection['parent_run_id']!=parent:raise ValueError('Selection/run mismatch')
    model_path=folder/'selected_validation_model.joblib'
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    test_summary=json.loads((folder/'test/summary.json').read_text())
    if test_summary['model_sha256']!=model_hash:raise ValueError('Model changed since final evaluation')
    model=joblib.load(model_path)
    data_path=ROOT/'data/processed/prepared.parquet'
    readiness=inspect_data(data_path)
    data=pd.read_parquet(data_path);data['date']=pd.to_datetime(data.date)
    saved=pd.read_csv(folder/'selected_validation_predictions.csv',parse_dates=['date'])
    inputs=reconstruct_inputs(data,saved)
    x=features(inputs)
    expected=inputs[['id','date']].merge(saved[['id','date','prediction']],on=['id','date'],validate='one_to_one').prediction.to_numpy()
    if not np.allclose(np.maximum(model.predict(x),0),expected,rtol=1e-6,atol=1e-6):
        raise ValueError('Reconstructed inputs do not reproduce saved validation forecasts')
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'tracking/mlflow.db').as_posix());mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='frozen_model_explainability',tags={'phase':'explainability','development_run_id':parent}) as run:
        out=folder/'xai'/run.info.run_id;out.mkdir(parents=True)
        mlflow.log_params({'model_sha256':model_hash,'data_sha256':readiness['sha256'],
                          'explanation_rows':len(x),'explanation_period':'validation',
                          'method':'TreeSHAP_tree_path_dependent','model_refitted':False})
        mlflow.log_artifact(__file__,artifact_path='code')
        print('Computing SHAP contributions for the 2,800 validation forecasts...',flush=True)
        values,base,raw=grouped_shap(model,x)
        importance=pd.DataFrame({'feature':FEATURES,'mean_absolute_shap':np.abs(values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False)
        importance.to_csv(out/'importance.csv',index=False)
        np.savez_compressed(out/'shap.npz',values=values,base=base,raw=raw,features=np.array(FEATURES))
        inputs[['id','date']+FEATURES].to_csv(out/'inputs.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,5.5));ordered=importance.iloc[::-1]
        ax.barh(ordered.feature,ordered.mean_absolute_shap,color='#008b8b')
        ax.set_xlabel('Mean absolute SHAP contribution (daily sales units)')
        ax.set_title('Inputs influencing validation forecasts');fig.tight_layout();fig.savefig(out/'global.png',dpi=160);plt.close(fig)
        # Predeclared example: first forecast date, lexicographically first series.
        i=0
        local=pd.DataFrame({'feature':FEATURES,'input_value':x.iloc[i].astype(str).to_numpy(),'shap_contribution':values[i]})
        local.to_csv(out/'local.csv',index=False)
        ordered=local.assign(magnitude=local.shap_contribution.abs()).sort_values('magnitude').tail(10)
        fig,ax=plt.subplots(figsize=(9,5.5));ax.barh(ordered.feature,ordered.shap_contribution,color=np.where(ordered.shap_contribution>=0,'#008b8b','#d05050'))
        ax.axvline(0,color='gray',linewidth=.7);ax.set_xlabel('Contribution to raw predicted sales (units)')
        ax.set_title(f'{inputs.iloc[i].id} | {inputs.iloc[i].date.date()}\nBase {base[i]:.3f} + all contributions = raw prediction {raw[i]:.3f}')
        fig.tight_layout();fig.savefig(out/'local.png',dpi=160);plt.close(fig)
        summary={'xai_run_id':run.info.run_id,'development_run_id':parent,'model_sha256':model_hash,
                 'explanation_rows':len(x),'series':inputs.id.nunique(),'period':'2016-03-28 to 2016-04-24',
                 'method':'TreeSHAP, tree_path_dependent; training tree path counts supply the reference distribution.',
                 'global_definition':'Mean absolute grouped SHAP contributions across all validation forecasts.',
                 'top_features':importance.head(5).to_dict('records'),
                 'local_example':{'id':str(inputs.iloc[i].id),'date':str(inputs.iloc[i].date.date()),
                                  'base_value':float(base[i]),'raw_prediction':float(raw[i]),'served_prediction':float(max(raw[i],0))},
                 'max_additivity_error':float(np.max(np.abs(base+values.sum(axis=1)-raw))),
                 'negative_raw_predictions':int((raw<0).sum()),
                 'limitations':['Explains raw model output; forecasts clamp negative output to zero.',
                                'Correlated lags and one-hot category grouping affect attribution.',
                                'These associations are not causal effects or stock-order recommendations.',
                                'Scope is the fixed 100-series cohort and validation period; no model retuning.']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        for path in out.iterdir():mlflow.log_artifact(str(path))
        print('\nSHAP FEATURE IMPORTANCE\n'+importance.to_string(index=False))
        print('\n'+json.dumps(summary,indent=2));print('\nExplanation files saved in:',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run',required=True)
    args=parser.parse_args()
    if len(args.parent_run)!=32 or any(c not in '0123456789abcdef' for c in args.parent_run):parser.error('Invalid development run ID')
    run_analysis(args.parent_run)


if __name__=='__main__':main()
