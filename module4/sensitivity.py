"""Evaluate fixed-model input sensitivity on validation forecast snapshots."""
from __future__ import annotations
import argparse
import hashlib
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from module4.setup_tracking import ROOT, EXPERIMENT, inspect_data
from module4.train_models import features, TRAIN_END
from module4.explain_model import reconstruct_inputs

HISTORY_FEATURES=['lag_7','lag_28','rolling_mean_28']
SCENARIOS=[('history',-.20),('history',-.10),('history',.10),('history',.20),
           ('price',-.10),('price',.10)]


def perturb(x,kind,change):
    if kind not in ['history','price'] or change<=-1:raise ValueError('Invalid scenario')
    result=x.copy()
    cols=HISTORY_FEATURES if kind=='history' else ['historical_sell_price']
    result[cols]=result[cols]*(1+change)
    return result


def sensitivity_table(model,x,ids,train):
    baseline=np.maximum(model.predict(x),0)
    records=[];row_outputs=[]
    for kind,change in SCENARIOS:
        altered=perturb(x,kind,change)
        pred=np.maximum(model.predict(altered),0)
        delta=pred-baseline
        cols=HISTORY_FEATURES if kind=='history' else ['historical_sell_price']
        # Range screen uses only observed training input values for each series.
        lo=train.groupby('id')[cols].min().reindex(ids).to_numpy()
        hi=train.groupby('id')[cols].max().reindex(ids).to_numpy()
        outside=((altered[cols].to_numpy()<lo)|(altered[cols].to_numpy()>hi)).any(axis=1)
        name=f'{kind}_{change:+.0%}'
        records.append({'scenario':name,'input_change_percent':100*change,
                        'baseline_mean_prediction':float(baseline.mean()),
                        'scenario_mean_prediction':float(pred.mean()),
                        'mean_prediction_change_units':float(delta.mean()),
                        'aggregate_prediction_change_percent':float(100*delta.sum()/baseline.sum()) if baseline.sum()>0 else None,
                        'mean_absolute_change_units':float(np.abs(delta).mean()),
                        'prediction_change_p10':float(np.quantile(delta,.1)),
                        'prediction_change_p90':float(np.quantile(delta,.9)),
                        'outside_training_range_fraction':float(outside.mean())})
        row_outputs.append(pd.DataFrame({'id':np.asarray(ids),'scenario':name,
             'baseline_prediction':baseline,'scenario_prediction':pred,'prediction_change':delta,
             'outside_training_range':outside}))
    return pd.DataFrame(records),pd.concat(row_outputs,ignore_index=True)


def run_analysis(parent,xai_run):
    folder=ROOT/'models/module4'/parent
    xai=folder/'xai'/xai_run
    meta=json.loads((xai/'summary.json').read_text())
    model_path=folder/'selected_validation_model.joblib'
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if meta['model_sha256']!=digest or meta['development_run_id']!=parent:raise ValueError('Explanation/model mismatch')
    model=joblib.load(model_path)
    source=ROOT/'data/processed/prepared.parquet'
    readiness=inspect_data(source)
    data=pd.read_parquet(source);data['date']=pd.to_datetime(data.date)
    train=data[data.date<=TRAIN_END]
    saved=pd.read_csv(folder/'selected_validation_predictions.csv',parse_dates=['date'])
    # Rebuild from the original Parquet/calendar rather than CSV-converted inputs.
    # This preserves categorical values and floating-point precision at tree splits.
    inputs=reconstruct_inputs(data,saved)
    x=features(inputs)
    expected=inputs[['id','date']].merge(saved[['id','date','prediction']],on=['id','date'],validate='one_to_one').prediction.to_numpy()
    if not np.allclose(np.maximum(model.predict(x),0),expected,rtol=1e-6,atol=1e-6):raise ValueError('Baseline does not reproduce saved validation predictions')
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'tracking/mlflow.db').as_posix());mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='frozen_model_sensitivity',tags={'phase':'sensitivity','development_run_id':parent}) as run:
        out=folder/'sens'/run.info.run_id;out.mkdir(parents=True)
        mlflow.log_params({'model_sha256':digest,'data_sha256':readiness['sha256'],'rows':len(x),
                           'period':'validation','model_refitted':False,'analysis':'snapshot_input_perturbation'})
        mlflow.log_artifact(__file__,artifact_path='code')
        table,rows=sensitivity_table(model,x,inputs.id,train)
        table.to_csv(out/'scores.csv',index=False)
        rows.to_csv(out/'rows.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(table.scenario,table.mean_prediction_change_units,color=np.where(table.mean_prediction_change_units>=0,'#008b8b','#d05050'))
        ax.axvline(0,color='gray',linewidth=.8);ax.set_xlabel('Mean change in predicted daily sales (units)')
        ax.set_title('Frozen-model sensitivity to hypothetical inputs')
        fig.tight_layout();fig.savefig(out/'sens.png',dpi=160);plt.close(fig)
        summary={'sensitivity_run_id':run.info.run_id,'development_run_id':parent,'xai_run_id':xai_run,
                 'model_sha256':digest,'rows':len(x),'period':'validation','scenarios':table.to_dict('records'),
                 'history_policy':'Scale lag_7, lag_28 and rolling_mean_28 together; leave IDs, calendar and prices fixed.',
                 'price_policy':'Scale historical_sell_price only; leave all other inputs fixed.',
                 'limitations':['One-step sensitivity at saved forecast snapshots; does not regenerate a full recursive horizon.',
                                'Input scenarios are hypothetical and are not causal price elasticity or actionable sales-history changes.',
                                'Training min/max screen only detects extrapolation; it does not prove scenario plausibility.',
                                'No model fit, retuning or final-test evaluation; sensitivity is not a counterfactual search.']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        for path in out.iterdir():mlflow.log_artifact(str(path))
        print('\nSENSITIVITY COMPARISON\n'+table[['scenario','mean_prediction_change_units','aggregate_prediction_change_percent','outside_training_range_fraction']].to_string(index=False))
        print('\nResults saved in:',out)
        print('Model unchanged. Scenarios describe model responses, not real-world causal effects.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run',required=True);parser.add_argument('--xai-run',required=True)
    args=parser.parse_args()
    for value in [args.parent_run,args.xai_run]:
        if len(value)!=32 or any(c not in '0123456789abcdef' for c in value):parser.error('Invalid run ID')
    run_analysis(args.parent_run,args.xai_run)


if __name__=='__main__':main()
