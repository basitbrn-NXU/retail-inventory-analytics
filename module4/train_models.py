"""Track baseline, time-aware tuning and validation; do not evaluate final test data."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from module4.setup_tracking import ROOT, EXPERIMENT, inspect_data

TRAIN_END = pd.Timestamp('2016-03-27')
VALID_END = pd.Timestamp('2016-04-24')
FOLDS = [pd.Timestamp('2016-01-31'), pd.Timestamp('2016-02-28')]
CATEGORICAL = ['item_id', 'store_id', 'dept_id', 'state_id', 'event_name_1', 'event_name_2']
NUMERIC = ['wday', 'month', 'snap', 'lag_7', 'lag_28', 'rolling_mean_28', 'historical_sell_price']
FEATURES = CATEGORICAL + NUMERIC
KNOWN_FUTURE = ['id', 'date', 'item_id', 'store_id', 'dept_id', 'state_id',
                'event_name_1', 'event_name_2', 'wday', 'month', 'snap']
# Coarse grid declared before evaluation. Each family is tuned within training data only.
CANDIDATES = {
    'random_forest': [dict(n_estimators=80, max_depth=12, min_samples_leaf=5),
                      dict(n_estimators=80, max_depth=18, min_samples_leaf=10)],
    'gradient_boosting': [dict(max_iter=160, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0),
                          dict(max_iter=160, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=5.0)]}

def features(frame):
    x = frame[FEATURES].copy()
    for col in CATEGORICAL:
        x[col] = x[col].fillna('No_event').astype(str)
    if not np.isfinite(x[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError('Missing/nonfinite historical model inputs; do not impute using future labels.')
    return x

def build_model(family, params):
    encoding = ColumnTransformer([
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), CATEGORICAL),
        ('numeric', 'passthrough', NUMERIC)], verbose_feature_names_out=True)
    if family == 'random_forest':
        model = RandomForestRegressor(**params, random_state=42, n_jobs=2)
    elif family == 'gradient_boosting':
        # Disable default random validation for early stopping in time-series work.
        model = HistGradientBoostingRegressor(**params, random_state=42, early_stopping=False)
    else:
        raise ValueError(f'Unknown family: {family}')
    return Pipeline([('encoding', encoding), ('model', model)])

def forecast(model, history, future):
    """Forecast recursively from history; future inputs contain calendar/IDs only."""
    history = history.sort_values(['date', 'id'])
    future = future[KNOWN_FUTURE].sort_values(['date', 'id']).copy()
    ids = sorted(history.id.unique())
    dates = sorted(future.date.unique())
    origin = history.date.max()
    if len(dates) != 28 or pd.Timestamp(dates[0]) != origin + pd.Timedelta(days=1):
        raise ValueError('Forecast must start immediately after history and span 28 days.')
    expected = list(pd.date_range(origin + pd.Timedelta(days=1), periods=28))
    if list(pd.to_datetime(dates)) != expected:
        raise ValueError('Forecast calendar must be consecutive.')
    hist = history.pivot(index='date', columns='id', values='units').reindex(columns=ids)
    if len(hist) < 28 or hist.tail(28).isna().any().any():
        raise ValueError('Each history needs 28 complete consecutive days.')
    if not hist.tail(28).index.equals(pd.date_range(origin-pd.Timedelta(days=27), origin, name='date')):
        raise ValueError('Recent history has date gaps.')
    values = hist.tail(28).to_numpy(dtype=float)
    prices = history.sort_values('date').groupby('id')['historical_sell_price'].last().reindex(ids).to_numpy()
    outputs = []
    for day in dates:
        frame = future[future.date.eq(day)].set_index('id').reindex(ids).reset_index()
        if len(frame) != len(ids) or frame.date.isna().any():
            raise ValueError('Future calendar must contain every selected history.')
        frame['lag_7'] = values[-7]
        frame['lag_28'] = values[-28]
        frame['rolling_mean_28'] = values[-28:].mean(axis=0)
        frame['historical_sell_price'] = prices
        pred = values[-7].copy() if model is None else np.maximum(model.predict(features(frame)), 0)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite predictions')
        outputs.append(frame[['id', 'date', 'store_id', 'dept_id']].assign(prediction=pred))
        values = np.vstack([values[-27:], pred])
    return pd.concat(outputs, ignore_index=True)

def score(actual, prediction):
    y = np.asarray(actual, dtype=float); p = np.asarray(prediction, dtype=float)
    out = {'mae': float(mean_absolute_error(y,p)), 'rmse': float(np.sqrt(mean_squared_error(y,p))),
           'r2': float(r2_score(y,p)), 'mean_signed_error': float(np.mean(p-y)),
           'underprediction_rate': float(np.mean(p<y))}
    if y.sum()>0:
        out['wape'] = float(np.abs(y-p).sum()/y.sum())
        out['aggregate_bias_fraction'] = float((p-y).sum()/y.sum())
    return out

def evaluate(pred, labels):
    joined = pred.merge(labels[['id','date','units']], on=['id','date'], validate='one_to_one')
    if len(joined) != len(pred): raise ValueError('Prediction/target mismatch')
    return score(joined.units, joined.prediction), joined

def group_scores(joined):
    records=[]
    for dimension in ['store_id','dept_id']:
        for name, group in joined.groupby(dimension):
            records.append({'dimension': dimension, 'group': str(name), 'rows': len(group),
                            'series': group.id.nunique(), **score(group.units,group.prediction)})
    return pd.DataFrame(records)

def save_csv(frame, path):
    frame.to_csv(path,index=False)
    mlflow.log_artifact(str(path))

def logged_fit(family, params, train, future, labels, phase, name, folder, data_hash):
    """Every fit begins inside a new MLflow run; fit order follows time."""
    with mlflow.start_run(run_name=name, nested=True, tags={'phase':phase,'algorithm':family}) as run:
        mlflow.log_params({**params, 'seed':42, 'train_end':str(train.date.max().date()),
                           'forecast_days':28, 'train_rows':len(train), 'data_sha256':data_hash,
                           'encoding':'one_hot', 'prediction_strategy':'recursive',
                           'future_price_policy':'last_observed', 'early_stopping':False})
        print(f'  Fitting {name}: {len(train):,} training rows', flush=True)
        model=build_model(family,params)
        x=features(train)
        with threadpool_limits(limits=2):
            model.fit(x,train.units)
            pred=forecast(model,train,future)
        metrics, joined=evaluate(pred,labels)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', signature=mlflow.models.infer_signature(x.head(5),model.predict(x.head(5))))
        save_csv(joined,folder/f'{name}_predictions.csv')
        print(f'  Finished {name}: MAE={metrics["mae"]:.4f}, RMSE={metrics["rmse"]:.4f}',flush=True)
        return model,metrics,joined,run.info.run_id

def run_experiment(data_path, output_root):
    readiness=inspect_data(data_path)
    df=pd.read_parquet(data_path)
    df['date']=pd.to_datetime(df.date)
    # Discard test rows before constructing any experiment input or target array.
    development=df.loc[df['split'].isin(['train','validation'])].copy()
    del df
    train=development.loc[development.date<=TRAIN_END].copy()
    valid=development.loc[(development.date>TRAIN_END)&(development.date<=VALID_END)].copy()
    future=valid[KNOWN_FUTURE].copy()
    if train.date.max()!=TRAIN_END or valid.date.max()!=VALID_END:
        raise ValueError('Unexpected development dates')
    uri='sqlite:///'+(ROOT/'tracking/mlflow.db').as_posix()
    if not (ROOT/'tracking/mlflow.db').exists():raise ValueError('Run module4.setup_tracking first.')
    mlflow.set_tracking_uri(uri);mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name='forecast_model_development', tags={'phase':'development','test_evaluated':'false'}) as parent:
        output=output_root/parent.info.run_id;output.mkdir(parents=True,exist_ok=False)
        protocol={'target':'recorded daily sales units','horizon_days':28,'seed':42,
          'fold_origins':[str(x.date()) for x in FOLDS], 'validation_origin':str(TRAIN_END.date()),
          'validation_end':str(VALID_END.date()),'selection_metric':'validation RMSE after CV tuning within each algorithm',
          'cv_metric':'mean fold RMSE','candidate_grid':CANDIDATES,
          'acceptance_targets':{'relative_rmse_improvement_over_weekly_baseline':0.05,
                                'absolute_aggregate_bias_fraction_max':0.10},
          'acceptance_status':'targets set before training; operational deployment also requires later test, XAI and group review',
          'scope':'fixed 100-series Module 2 cohort; CV conditional on cohort chosen using the full training period',
          'price_assumption':'last observed price held constant through each horizon',
          'calendar_assumption':'events and SNAP calendars available at forecast origin',
          'features':FEATURES,'test_evaluated':False,'data_sha256':readiness['sha256']}
        mlflow.log_dict(protocol,'protocol.json')
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        mlflow.log_artifact(__file__,artifact_path='code')
        mlflow.log_params({'data_sha256':readiness['sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'scikit_learn_version':importlib.metadata.version('scikit-learn')})
        cv=[];summary=[];models={};validation_outputs={}
        with mlflow.start_run(run_name='weekly_baseline_validation',nested=True,tags={'phase':'validation','algorithm':'weekly_baseline'}):
            baseline,joined=evaluate(forecast(None,train,future),valid)
            mlflow.log_params({'season_length_days':7,'horizon_days':28})
            mlflow.log_metrics(baseline);save_csv(joined,output/'weekly_baseline_validation_predictions.csv')
            summary.append({'model':'weekly_baseline',**baseline})
        for family,grid in CANDIDATES.items():
            candidates=[]
            for index,params in enumerate(grid):
                scores=[]
                for fold,origin in enumerate(FOLDS):
                    fit=train[train.date<=origin].copy()
                    holdout=train[(train.date>origin)&(train.date<=origin+pd.Timedelta(days=28))].copy()
                    name=f'{family}_config{index+1}_fold{fold+1}'
                    _,metrics,_,rid=logged_fit(family,params,fit,holdout[KNOWN_FUTURE],holdout,'cv',name,output,readiness['sha256'])
                    scores.append(metrics['rmse']);cv.append({'model':family,'config':index+1,'fold':fold+1,'origin':str(origin.date()),'run_id':rid,**metrics})
                candidates.append((float(np.mean(scores)),index,params))
            best_cv,index,params=min(candidates,key=lambda x:(x[0],x[1]))
            model,metrics,joined,rid=logged_fit(family,params,train,future,valid,'validation',family+'_validation',output,readiness['sha256'])
            models[family]=model;validation_outputs[family]=joined
            summary.append({'model':family,'config':index+1,'cv_rmse':best_cv,'run_id':rid,**metrics})
            save_csv(group_scores(joined),output/f'{family}_validation_groups.csv')
        table=pd.DataFrame(summary).sort_values('rmse');save_csv(table,output/'validation_comparison.csv')
        save_csv(pd.DataFrame(cv),output/'cross_validation.csv')
        # Keep the best learned model for explainability, even if the baseline remains better.
        best=min((row for row in summary if row['model']!='weekly_baseline'),key=lambda row:row['rmse'])
        improvement=1-best['rmse']/baseline['rmse'] if baseline['rmse']>0 else None
        decision={'best_learned_model':best['model'],'validation_winner_including_baseline':table.iloc[0]['model'],
                  'source_run_id':best['run_id'],'parent_run_id':parent.info.run_id,
                  'relative_rmse_improvement':improvement,
                  'passes_provisional_accuracy_gate':bool(improvement is not None and improvement>=0.05 and abs(best.get('aggregate_bias_fraction',float('inf')))<=0.10),
                  'final_test_evaluated':False,'deployment_approved':False,
                  'next':'freeze model choice, evaluate final test, then complete XAI, group review, Model Card and API'}
        model_path=output/'selected_validation_model.joblib'
        joblib.dump(models[best['model']],model_path,compress=3);mlflow.log_artifact(str(model_path))
        (output/'selection.json').write_text(json.dumps(decision,indent=2));mlflow.log_dict(decision,'selection.json')
        save_csv(validation_outputs[best['model']],output/'selected_validation_predictions.csv')
        print('\nVALIDATION COMPARISON\n'+table[['model','mae','rmse','mean_signed_error']].to_string(index=False),flush=True)
        print('\n'+json.dumps(decision,indent=2),flush=True)
        print('\nResults saved in:',output,flush=True)
        print('Refresh MLflow Model Training to view runs. Final test data has not been evaluated.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed/prepared.parquet')
    args=parser.parse_args()
    run_experiment(args.data,ROOT/'models/module4')

if __name__=='__main__':
    main()
