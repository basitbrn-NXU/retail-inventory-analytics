"""DiCE counterfactual search over coherent alternative sales-history inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import dice_ml
from module4.setup_tracking import ROOT,EXPERIMENT,inspect_data
from module4.train_models import features,TRAIN_END
from module4.explain_model import reconstruct_inputs

HISTORY=['lag_7','lag_28','rolling_mean_28']

class HistoryScaleModel:
    """Expose one scenario parameter; call the existing model without fitting."""
    def __init__(self,model,query):self.model=model;self.query=query.copy()
    def predict(self,scenarios):
        scales=np.asarray(scenarios['history_scale'],dtype=float)
        expanded=pd.concat([self.query]*len(scales),ignore_index=True)
        expanded[HISTORY]=expanded[HISTORY].to_numpy()*scales[:,None]
        return np.maximum(self.model.predict(expanded),0)


def search(model,query,series_train):
    wrapper=HistoryScaleModel(model,query)
    base=float(wrapper.predict(pd.DataFrame({'history_scale':[1.]}))[0])
    target=[base*1.10,base*1.20]
    # Uniform scaling represents one alternative 28-day history, keeping lags consistent.
    lower,upper=.5,1.5
    for col in HISTORY:
        value=float(query.iloc[0][col])
        if value>0:
            lower=max(lower,float(series_train[col].min())/value)
            upper=min(upper,float(series_train[col].max())/value)
    info={'baseline_prediction':base,'desired_prediction_range':target,
          'permitted_history_scale':[lower,upper], 'method':'DiCE random search',
          'seed':42,'sample_size':5000,'found':False}
    if base<=0 or lower>1 or upper<1 or upper-lower<1e-6:
        info['reason']='The fixed target or training-range constraints do not allow this search.'
        return info,pd.DataFrame()
    # DiCE's data interface describes the scenario parameter space, not real sales records.
    grid=pd.DataFrame({'history_scale':np.linspace(lower,upper,501)})
    grid['predicted_units']=wrapper.predict(grid)
    interface=dice_ml.Data(dataframe=grid,continuous_features=['history_scale'],
                           outcome_name='predicted_units',continuous_features_precision={'history_scale':6})
    predictor=dice_ml.Model(model=wrapper,backend='sklearn',model_type='regressor')
    explainer=dice_ml.Dice(interface,predictor,method='random')
    try:
        result=explainer.generate_counterfactuals(pd.DataFrame({'history_scale':[1.]}),
              total_CFs=3,desired_range=target,features_to_vary=['history_scale'],
              permitted_range={'history_scale':[lower,upper]},random_seed=42,
              sample_size=5000,posthoc_sparsity_param=0)
    except Exception as error:
        # DiCE raises this specific error when its finite random search finds no solutions.
        if 'No counterfactuals found' not in str(error):raise
        info['reason']='DiCE found no solution within the declared range and search budget; target was not relaxed.'
        return info,pd.DataFrame()
    candidates=result.cf_examples_list[0].final_cfs_df
    if candidates is None or candidates.empty:
        info['reason']='DiCE found no solution within the declared constraints.'
        return info,pd.DataFrame()
    candidates=candidates.copy()
    candidates['verified_prediction']=wrapper.predict(candidates)
    candidates['distance_from_original_scale']=abs(candidates.history_scale-1)
    candidates=candidates.sort_values('distance_from_original_scale')
    valid=candidates.verified_prediction.between(target[0]-1e-6,target[1]+1e-6)
    candidates=candidates.loc[valid].copy()
    if candidates.empty:raise ValueError('Returned counterfactuals did not reproduce the requested outcome')
    for col in HISTORY:
        candidates['original_'+col]=float(query.iloc[0][col])
        candidates['alternative_'+col]=candidates.history_scale*float(query.iloc[0][col])
    info['found']=True
    info['solutions']=len(candidates)
    info['nearest_found_scale']=float(candidates.iloc[0].history_scale)
    info['nearest_found_prediction']=float(candidates.iloc[0].verified_prediction)
    info['optimality']='Nearest among returned solutions only; random search does not guarantee a minimum change.'
    return info,candidates


def run_analysis(parent):
    folder=ROOT/'models/module4'/parent
    model_path=folder/'selected_validation_model.joblib';digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    test=json.loads((folder/'test/summary.json').read_text())
    if test['model_sha256']!=digest:raise ValueError('Model differs from the evaluated model')
    model=joblib.load(model_path)
    source=ROOT/'data/processed/prepared.parquet';ready=inspect_data(source)
    data=pd.read_parquet(source);data['date']=pd.to_datetime(data.date)
    saved=pd.read_csv(folder/'selected_validation_predictions.csv',parse_dates=['date'])
    inputs=reconstruct_inputs(data,saved)
    # Same example used in local SHAP; chosen by date/series, not forecast outcome.
    query=features(inputs).iloc[[0]]
    key=str(inputs.iloc[0].id)
    train=data[(data.date<=TRAIN_END)&data.id.eq(key)]
    baseline=float(max(model.predict(query)[0],0))
    expected=float(saved[(saved.id.eq(key))&saved.date.eq(inputs.iloc[0].date)].prediction.iloc[0])
    if not np.isclose(baseline,expected,rtol=1e-6,atol=1e-6):raise ValueError('Query does not reproduce the saved forecast')
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'tracking/mlflow.db').as_posix());mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='frozen_model_counterfactual',tags={'phase':'counterfactual','development_run_id':parent}) as run:
        out=folder/'cf'/run.info.run_id;out.mkdir(parents=True)
        mlflow.log_params({'model_sha256':digest,'data_sha256':ready['sha256'],'seed':42,
                           'method':'DiCE_random','target_increase_min':.10,'target_increase_max':.20,'model_refitted':False})
        mlflow.log_artifact(__file__,artifact_path='code')
        print('Searching for alternative history inputs for the local SHAP example...',flush=True)
        info,candidates=search(model,query,train)
        summary={**info,'counterfactual_run_id':run.info.run_id,'development_run_id':parent,
                 'id':key,'date':str(inputs.iloc[0].date.date()),'model_sha256':digest,
                 'fixed_inputs':'Product/store/department/state IDs, calendar, SNAP and price are unchanged.',
                 'history_rule':'Scale lag_7, lag_28 and rolling_mean_28 by the same factor, bounded by 0.5–1.5 and this series training min/max.',
                 'limitations':['Alternative history is an explanatory scenario, not actionable recourse; past sales cannot be changed.',
                                'Range constraints do not prove realistic customer behaviour or causal effects.',
                                'The DiCE interface grid represents hypothetical scale settings; no new model is fitted.',
                                'Finite random search does not guarantee global minimality or infeasibility when no solution is found.',
                                'No retuning or repeated final-test evaluation.']}
        query.to_csv(out/'original.csv',index=False)
        candidates.to_csv(out/'alternatives.csv',index=False)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        for path in out.iterdir():mlflow.log_artifact(str(path))
        print('\nCOUNTERFACTUAL RESULT\n'+json.dumps(summary,indent=2))
        if not candidates.empty:print('\n'+candidates.to_string(index=False))
        print('\nResults saved in:',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-run',required=True)
    args=parser.parse_args()
    if len(args.parent_run)!=32 or any(c not in '0123456789abcdef' for c in args.parent_run):parser.error('Invalid run ID')
    run_analysis(args.parent_run)


if __name__=='__main__':main()
